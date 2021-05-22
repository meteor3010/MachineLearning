﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    /// <summary>
    /// Works
    /// </summary>
    public class SinglePerceptron
    {
        public int NumberInput;

        const double m_LearningRate = 0.1;

        List<double> weights;
        double bias;
        public SinglePerceptron(int numberInput)
        {
            NumberInput = numberInput;
            Random random = new Random(Guid.NewGuid().GetHashCode());

            weights = new List<double>();

            for (int i = 0; i < NumberInput; i++)
            {
                weights.Add(new Random(Guid.NewGuid().GetHashCode()).NextDouble() / 10);
            }
            bias = random.NextDouble();
        }

        public double Predict(List<double> datas)
        {
            double result = 0;
            for (int i = 0; i < datas.Count; i++)
            {
                result += weights[i] * datas[i] + bias;
            }

            return result;
        }

        public void Train(List<double> datas, double result)
        {
            double guess = Predict(datas);

            if (Sign(guess) != result)
            {
                for (int i = 0; i < weights.Count; i++)
                {
                    weights[i] += m_LearningRate * (result - guess) * datas[i];
                }

                bias += m_LearningRate * (result - guess);
            }

        }

        private static double Sign(double number)
        {
            if (number > 0)
                return 1;

            return -1;
        }
    }
}