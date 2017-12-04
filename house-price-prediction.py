import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as mpl
import matplotlib.animation as animation

class HousePricePredictor():

    def run(self):
        num_houses = 160
        size = self.get_house_sizes(num_houses)
        price = self.get_house_prices(size, num_houses)
        self.plot_house_size_by_price(size, price)

    def get_house_sizes(self, num_houses):
        np.random.seed(42)
        house_sizes = np.random.randint(low=1000, high=3500, size=num_houses)
        return house_sizes

    def get_house_prices(self, size, num_houses):
        np.random.seed(42)
        house_prices = size * 100.0 + np.random.randint(low=20000, high = 70000, size=num_houses)
        return house_prices

    def plot_house_size_by_price(self, size, price):
        mpl.plot(size, price, "bx")
        mpl.ylabel("Price")
        mpl.xlabel("Size")
        mpl.show()


if __name__ == "__main__":
    app = HousePricePredictor()
    app.run()

