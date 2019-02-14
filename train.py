import random
import numpy as np
import tensorflow as tf


NUM_AGENTS = 10
NUM_MATCHES = 1000
INPUT_LENGTH = 100
LATENT_DIMS = [100, 100]
BATCH_SIZE = 200

class Agent:

    def __init__(self, id):
        self.id = id

    def build_agent(self):
        with tf.variable_scope("agent_" + str(self.id)):

            self.x = tf.placeholder(shape=[BATCH_SIZE, INPUT_LENGTH], dtype=tf.float32)
            self.y = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.uint8)
            x = self.x
            for dim in LATENT_DIMS:
                x = tf.layers.dense(x, dim, activation=tf.nn.relu)

            self.logits = tf.layers.dense(x, 2)
            self.proba = tf.nn.softmax(self.logits)
            self.labels = tf.one_hot(self.y, depth=2)

            self.cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def predict(self):
        def eval_fn(x):
            return self.proba.eval(feed_dict={self.x: x})
        return eval_fn


agents = []
for i in range(NUM_AGENTS):
    agent = Agent(i)
    agent.build_agent()
    agents.append(agent)


def train_set(a1, a2, n=200):
    x = np.random.rand(n, INPUT_LENGTH)
    y = np.zeros(shape=(n))
    y = np.expand_dims(y, -1)
    return (x, y)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(NUM_MATCHES):
        a1, a2 = random.sample(agents, 2)
        x, y = train_set(a1, a2, BATCH_SIZE)

        a1_pred_fn = a1.predict()(x)
        a2_pred_fn = a2.predict()(x)

        print(preds)
        _, a1_cost = sess.run([a1.optimizer, a1.cost], feed_dict={a1.x: x, a1.y: y})
        print("a1_cost", a1_cost)

        """
        _, a1_cost, a2_cost = sess.run([a2.optimizer, a1.cost, a2.cost], feed_dict={a1.x: x, a1.y: y})
        print("a1_cost", a1_cost)
        print("a2_cost", a2_cost)
        """
    print(a1)
    print(a2)