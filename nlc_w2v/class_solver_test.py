import numpy
from unittest import TestCase
from .class_solver import ClassSolver


class ClassSolverTest(TestCase):
    def setUp(self):
        self.solver = ClassSolver(2)

    def testTrain(self):
        def real_func(x):
            return {True: (1.0, 0.0,), False: (0.0, 1.0,)}[x > 0.5]
        x = []
        y = []
        for i in range(0, 100):
            x_val = i / 100.0
            x.append([x_val, x_val])
            y1, y2 = real_func(x_val)
            y.append([y1, y2])
        self.solver.train(numpy.array(x), numpy.array(y), verbose=True)
        for i in range(0, 100):
            x_val = i / 100.0
            solved = self.solver.calculate(numpy.array([x_val, x_val]))
            y1_real, y2_real = real_func(x_val)
            y1_approx = solved[0]
            y2_approx = solved[1]
            print("f({0}) = [{1}, {2}], f_approx({0}) = [{3}, {4}]".format(
                x_val, y1_real, y2_real, y1_approx, y2_approx
            ))
            self.assertTrue(
                abs(y1_real - y1_approx) < 0.3
            )
            self.assertTrue(
                abs(y2_real - y2_approx) < 0.3
            )
