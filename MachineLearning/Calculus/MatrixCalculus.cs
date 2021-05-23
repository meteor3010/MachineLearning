﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Calculus
{
    public static class MatrixCalculus
    {
        public static List<List<double>> Identity(int n, int m)
        {
            List<List<double>> matrix = new List<List<double>>();

            for (int i = 0; i < n; i++)
            {
                List<double> row = new List<double>();
                matrix.Add(row);
                for (int j = 0; j < m; j++)
                    row.Add(1);
            }

            return matrix;
        }

        public static List<List<double>> Multiply(List<List<double>> matrix1, List<List<double>> matrix2)
        {
            List<List<double>> multiplication = new List<List<double>>();

            for (int i = 0; i < matrix1.Count; i++)
            {
                List<double> row = new List<double>();

                for (int j = 0; j < matrix2[0].Count; j++)
                {
                    double v = 0;

                    for (int k = 0; k < matrix2.Count; k++)
                        v += matrix1[i][k] * matrix2[k][j];

                    row.Add(v);
                }
                multiplication.Add(row);
            }

            return multiplication;
        }

        public static List<List<double>> HadamarProduct(List<List<double>> matrix1, List<List<double>> matrix2)
        {
            List<List<double>> addition = new List<List<double>>();

            for (int i = 0; i < matrix1.Count; i++)
            {
                List<double> row = new List<double>();
                for (int j = 0; j < matrix1[0].Count; j++)
                    row.Add(matrix1[i][j] * matrix2[i][j]);

                addition.Add(row);
            }

            return addition;
        }


        public static List<List<double>> Multiply(List<List<double>> matrix, double scalar)
        {
            List<List<double>> multiplication = new List<List<double>>();

            for (int i = 0; i < matrix.Count; i++)
            {
                List<double> row = new List<double>();

                for (int j = 0; j < matrix[0].Count; j++)
                    row.Add(matrix[i][j] * scalar);

                multiplication.Add(row);
            }

            return multiplication;
        }

        public static List<List<double>> Add(List<List<double>> matrix1, List<List<double>> matrix2)
        {
            List<List<double>> addition = new List<List<double>>();

            for (int i = 0; i < matrix1.Count; i++)
            {
                List<double> row = new List<double>();
                for (int j = 0; j < matrix1[0].Count; j++)
                    row.Add(matrix1[i][j] + matrix2[i][j]);

                addition.Add(row);
            }

            return addition;
        }

        public static List<List<double>> Substract(List<List<double>> matrix1, List<List<double>> matrix2)
        {
            List<List<double>> addition = new List<List<double>>();

            for (int i = 0; i < matrix1.Count; i++)
            {
                List<double> row = new List<double>();
                for (int j = 0; j < matrix1[0].Count; j++)
                    row.Add(matrix1[i][j] - matrix2[i][j]);

                addition.Add(row);
            }

            return addition;
        }

        public static List<List<double>> RiseValuesToSquare(List<List<double>> matrix)
        {
            List<List<double>> addition = new List<List<double>>();

            for (int i = 0; i < matrix.Count; i++)
            {
                List<double> row = new List<double>();

                for (int j = 0; j < matrix[0].Count; j++)
                    row.Add(Math.Pow(matrix[i][j], 2));

                addition.Add(row);
            }

            return addition;
        }

        public static List<List<double>> Transpose(List<List<double>> matrix)
        {
            List<List<double>> result = new List<List<double>>();
            for (int i = 0; i < matrix[0].Count; i++)
            {
                List<double> row = new List<double>();
                result.Add(row);
                for (int j = 0; j < matrix.Count; j++)
                {
                    row.Add(matrix[j][i]);
                }
            }
            return result;
        }

        public static List<List<double>> Sigmoid(List<List<double>> matrix)
        {
            List<List<double>> result = new List<List<double>>();

            for (int i = 0; i < matrix.Count; i++)
            {
                List<double> row = new List<double>();
                result.Add(row);
                for (int j = 0; j < matrix[0].Count; j++)
                    row.Add(Sigmoid(matrix[i][j]));
            }
            return result;
        }

        public static List<List<double>> Map(List<List<double>> matrix, Func<double, double> action)
        {
            List<List<double>> result = new List<List<double>>();

            for (int i = 0; i < matrix.Count; i++)
            {
                List<double> row = new List<double>();
                result.Add(row);

                for (int j = 0; j < matrix[0].Count; j++)
                    row.Add(action(matrix[i][j]));
            }

            return result;
        }

        public static List<List<double>> Sign(List<List<double>> matrix)
        {
            List<List<double>> result = new List<List<double>>();

            for (int i = 0; i < matrix.Count; i++)
            {
                List<double> row = new List<double>();
                result.Add(row);
                for (int j = 0; j < matrix[0].Count; j++)
                    row.Add(Sign(matrix[i][j]));
            }
            return result;
        }

        public static List<List<double>> GetMatrix(List<double> vector)
        {
            List<List<double>> expectedValues2 = new List<List<double>>();
            foreach (double value in vector)
                expectedValues2.Add(new List<double>() { value });

            return expectedValues2;
        }

        public static List<double> ToArray(List<List<double>> matrix)
        {
            List<double> output = new List<double>();

            foreach (List<double> row in matrix)
                output.Add(row[0]);

            return output;
        }

        private static double Sign(double number)
        {
            if (number > 0)
                return 1;

            return -1;
        }

        public static double Sigmoid(double number)
        {
            return 1.0 / (1 + Math.Exp(-number));
        }
    }
}