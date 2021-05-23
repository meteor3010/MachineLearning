﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Calculus
{
    public static class Tools
    {
        public static IEnumerable<IEnumerable<double>> ToMatrix(this IEnumerable<double> vector)
        {
            List<List<double>> expectedValues2 = new List<List<double>>();
            foreach (double value in vector)
                expectedValues2.Add(new List<double>() { value });

            return expectedValues2;
        }

        public static double Sigmoid(double number)
        {
            return 1.0 / (1 + Math.Exp(-number));
        }
    }
}