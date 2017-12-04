import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class HousePricePredictor():
    def __init__(self, data_size):
        self.num_houses = data_size
        self.num_samples = math.floor(self.num_houses * 0.7)
        self.house_sizes = self.randomize_house_sizes()
        self.house_prices = self.randomize_house_prices()


    def run(self):
        # self.plot_house_size_by_price(self.house_sizes, self.house_prices)
        train_house_sizes, train_house_prices = self.generate_train_data()

        train_house_size_norm = self.normalize(train_house_sizes)
        train_house_price_norm = self.normalize(train_house_prices)

        test_house_sizes, test_house_prices = self.generate_test_data()

        test_house_size_norm = self.normalize(test_house_sizes)
        test_house_price_norm = self.normalize(test_house_prices)

        tf_house_size, tf_price = self.assign_tf_placeholders()
        tf_size_factor, tf_price_offset = self.assign_tf_variables()

        tf_price_predictor = self.allocate_tf_operations(tf_size_factor, tf_house_size, tf_price_offset)
        tf_cost = self.allocate_tf_loss_function(tf_price_predictor, tf_price)
        optimizer = self.allocate_tf_gradient_descent_optimizer(tf_cost)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            display_every = 2
            num_training_inter = 50

            for iteration in range(num_training_inter):
                for(x, y) in zip(train_house_size_norm, train_house_price_norm):
                    sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

                    if(iteration + 1) % display_every == 0:
                        c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_house_price_norm})
                        print("interation #:" '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
                            "size_factor=", sess.run(tf_size_factor), "price_offset", sess.run(tf_price_offset))

            print("Optimization Finished")
            training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_house_price_norm})
            print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))


            # Plot of training and test data, and learned regression
    
            # get values used to normalized data so we can denormalize data back to its original scale
            train_house_size_mean = train_house_sizes.mean()
            train_house_size_std = train_house_sizes.std()

            train_price_mean = train_house_prices.mean()
            train_price_std = train_house_prices.std()

             # Plot the graph
            plt.rcParams["figure.figsize"] = (10,8)
            plt.figure()
            plt.ylabel("Price")
            plt.xlabel("Size (sq.ft)")
            plt.plot(train_house_sizes, train_house_prices, 'go', label='Training data')
            plt.plot(test_house_sizes, test_house_prices, 'mo', label='Testing data')
            plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
                    (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
                    label='Learned Regression')
        
            plt.legend(loc='upper left')
            plt.show()

            
    def randomize_house_sizes(self):
        np.random.seed(42)
        house_sizes = np.random.randint(low=1000, high=3500, size=self.num_houses)
        return house_sizes


    def randomize_house_prices(self):
        np.random.seed(42)
        house_prices = self.house_sizes * 100.0 + np.random.randint(low=20000, high = 70000, size=self.num_houses)
        return house_prices


    def plot_house_size_by_price(self, size, price):
        plt.plot(size, price, "bx")
        plt.ylabel("Price")
        plt.xlabel("Size")
        plt.show()


    def normalize(self, array):
        return (array - array.mean() / array.std())


    def generate_train_data(self):
        train_house_sizes = np.asarray(self.house_sizes[:self.num_samples ])
        train_house_prices = np.asarray(self.house_prices[:self.num_samples ])

        return train_house_sizes, train_house_prices


    def generate_test_data(self):
        test_house_sizes = np.asarray(self.house_sizes[:self.num_samples ])
        test_house_prices = np.asarray(self.house_prices[:self.num_samples ])

        return test_house_sizes, test_house_prices


    def assign_tf_placeholders(self):
        tf_house_size = tf.placeholder("float", name="house_size")
        tf_price = tf.placeholder("float", name="price")
        return tf_house_size, tf_price


    def assign_tf_variables(self):
        tf_size_factor = tf.Variable(np.random.randn(), name = "size_factor")
        tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")
        return tf_size_factor, tf_price_offset
    

    def allocate_tf_operations(self, tf_size_factor, tf_house_size, tf_price_offset):
        tf_price_predictor = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)
        return tf_price_predictor


    # Mean squared error
    def allocate_tf_loss_function(self, tf_price_predictor, tf_price):
        tf_cost = tf.reduce_sum(tf.pow(tf_price_predictor-tf_price, 2))/(2*self.num_samples)
        return tf_cost


    # Minimize the loss defined in operation "cost" 
    def allocate_tf_gradient_descent_optimizer(self, tf_cost):
        learning_rate = 0.1
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)


if __name__ == "__main__":
    num_houses = 160
    app = HousePricePredictor(num_houses)
    app.run()

