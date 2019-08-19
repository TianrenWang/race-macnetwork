import tensorflow as tf

# num_layers = 2
# d_model = 512
# dff = 512
# num_heads = 8
# dropout_rate = 0.1

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq # (batch_size, seq_len)

class Controller(tf.keras.layers.Layer):
    def __init__(self, d_model): #, num_layers, num_heads, dff, rate):
        super(Controller, self).__init__()

        self.cq = tf.keras.layers.Dense(d_model)
        self.focus = tf.keras.layers.Dense(1)
        # self.transformer = Transformer(num_layers, d_model, num_heads, dff, rate)


    def call(self, quest_state, control, context, training):
        # quest_state (the state of the question in each reasoning step): [batch, d]
        # control (the operation to be done this step): [batch, d]
        # context (BERT-encoded question): [batch, seq_len, d]

        cq = self.cq(tf.concat([quest_state, control])) #[batch, d]
        focus = self.focus(cq * context) # [batch, seq_len, d]
        focus = tf.keras.backend.squeeze(focus, 2)

        #attention, attention_weights = self.transformer(context, focus, training) #attention: [batch, seq_len, d]

        attention = tf.nn.softmax(focus) #[batch, seq_len]

        return tf.reduce_mean(attention * context, 1), attention # [batch, d]


class Reader(tf.keras.layers.Layer):
    def __init__(self, d_model): #, num_layers,  num_heads, dff, rate):
        super(Reader, self).__init__()

        self.memory = tf.keras.layers.Dense(d_model)
        self.knowledge = tf.keras.layers.Dense(d_model)
        self.disjoint = tf.keras.layers.Dense(d_model)
        self.retrieve = tf.keras.layers.Dense(1)
        #self.transformer = Transformer(num_layers, d_model, num_heads, dff, rate)

    def call(self, memory, knowledge, control, training):
        # memory (the memory of the previous reasoning step): [d]
        # control (the operation to be done this step): [d]
        # context (BERT-encoded question): [seq_len, d]

        reflection = self.memory(memory) * self.knowledge(knowledge)
        disjoint = self.disjoint(tf.concat([knowledge, reflection], 2))
        retrieve = self.retrieve(disjoint * control)
        retrieve = tf.keras.backend.squeeze(retrieve, 2)

        attention = tf.nn.softmax(retrieve) #[batch, seq_len]

        #attention, attention_weights = self.transformer(knowledge, retrieve, training)
        return tf.reduce_mean(attention * knowledge, 1), attention # [batch, d]


class Writer(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(Writer, self).__init__()

        self.m1 = tf.keras.layers.Dense(d_model)
        self.control_attention = tf.keras.layers.Dense(1)
        self.retrieve = tf.keras.layers.Dense(d_model)
        self.m2 = tf.keras.layers.Dense(d_model)
        self.s = tf.keras.layers.Dense(d_model)
        self.m3 = tf.keras.layers.Dense(1)

    def call(self, memory, read, control, past_results, training):
        # past_results (concatenation of all previous controls and memories): [batch, iterations, 2d]

        past_controls, past_memories = tf.split(past_results, 2, axis=2)

        m1 = self.m1(tf.concat([memory, read], 2))

        control_attention = self.control_attention(past_controls * control)
        control_attention = tf.squeeze(control_attention) # [batch, iterations]
        iteration_mask = create_padding_mask(control_attention)
        control_attention += iteration_mask * -1e9
        control_softmax = tf.keras.activations.softmax(control_attention)
        msa = tf.reduce_sum(control_softmax * past_memories, 1)

        mp = self.m2(m1) + self.s(msa)

        control_gate = tf.nn.sigmoid(self.m3(control))

        new_memory = mp * control_gate + (1 - control_gate) * memory

        return new_memory # [batch, d]

class Output(tf.keras.layers.Layer):
    def __init__(self, d_model, classes):
        super(Output, self).__init__()

        self.dense1 = tf.keras.layers.Dense(d_model)
        self.dense2 = tf.keras.layers.Dense(classes)

    def call(self, memory, quest_rep, training):
        # memory (the memory of the previous reasoning step): [batch, d]
        # quest_rep (self-attention on the BERT-encoded question): [batch, d]

        hidden1 = self.dense1(tf.concat([memory, quest_rep], 1))
        hidden1 = tf.nn.relu(hidden1)
        hidden2 = self.dense2(hidden1)

        return hidden2 # [batch, classes]

class MAC_Cell(tf.keras.layers.Layer):

    def __init__(self, d_model, steps): #, num_layers, num_heads, dff, rate):
        super(MAC_Cell, self).__init__()
        self.d_model = d_model
        self.controller = Controller(self.d_model) #, num_layers, num_heads, dff, rate)
        self.reader = Reader(self.d_model) #, num_layers, num_heads, dff, rate)
        self.write = Writer(self.d_model)
        self.question_state = tf.keras.layers.Dense(self.d_model)
        self.steps = steps


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        ones = tf.ones([batch_size, 1, self.d_model * 2])
        zeros = tf.zeros([batch_size, self.steps - 1, self.d_model * 2])
        return tf.concat([ones, zeros], 1) # [batch, steps, d_model * 2]

    @property
    def state_size(self):
        return [self.steps, self.d_model * 2]

    @property
    def output_size(self):
        return [self.steps, self.d_model * 2]


    def call(self, x, h, constants, training):
        # h (the control and memory of all the previous steps, respectively): [batch, steps, 2d]
        # x (the temporal encoding of the reasoning step): [batch, d]

        previous_state = tf.slice(h, [0, 0 , 0], [-1, 1, -1])
        prev_control, prev_memory = tf.split(previous_state, 2) # [batch, 1, d]
        prev_control = tf.squeeze(prev_control) # [batch, d]
        prev_memory = tf.squeeze(prev_memory) # [batch, d]

        knowledge = constants[0]
        question = constants[1]
        question_rep = constants[2]

        print("MAC_CELL X: " + str(tf.shape(x)))

        quest_state = self.question_state(x * question_rep)

        new_control, control_attention = self.controller(quest_state, prev_control, question, training)
        read, read_attention = self.reader(prev_memory, knowledge, new_control, training)
        new_memory = self.writer(prev_memory, read, new_control, h, training)

        new_state = tf.concat([new_control, new_memory])
        h = tf.slice(h, [0, 0, 0], [-1, h.shape[1] - 1, -1])
        h = tf.concat([tf.expand_dims(new_state, 1), h], 1)

        return h, h