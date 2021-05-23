using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
	class Program
	{
		static void Main(string[] args)
		{
			int nInput = 2;
			int nOutput = 1;
			int neuronsPerLayers = 20;
			int nLayers = 3;

			DeepLearning slp = new DeepLearning(nInput, neuronsPerLayers, nOutput, nLayers);
			//SingleLayerPerceptron slp = new SingleLayerPerceptron(nInput, neuronsPerLayers, nOutput);

			var nPoints = 100000;

			for (int i = 0; i < nPoints; i++)
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				int value1 = random.Next() % 2;
				random = new Random(Guid.NewGuid().GetHashCode());
				int value2 = random.Next() % 2;

				int xor = XOR(value1, value2);

				slp.Train(new List<double>() { value1, value2 }, new List<double>() { xor });
				var percent = Math.Round(i * 100.0 / nPoints, 4);
				//if (percent == (int)percent)
				//{
				//	Console.Clear();
				//	Console.WriteLine(percent);
				//}
			}

			double n = 0;
			int nTestCase = 200;
			for (int i = 0; i < nTestCase; i++)
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				int value1 = random.Next() % 2;
				random = new Random(Guid.NewGuid().GetHashCode());
				int value2 = random.Next() % 2;

				int xor = XOR(value1, value2);

				var guess = slp.Predict(new List<double>() { value1, value2 });
				double result = guess[0] > 0.5 ? 1 : 0;
				bool correct = result == xor;
				Console.WriteLine("xor : " + value1 + " X " + value2);
				Console.WriteLine("guess : " + guess[0]);
				Console.WriteLine("--");
				if (correct)
				{
					n++;
				}
			}

			//to debug
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				int value1 = random.Next() % 2;
				random = new Random(Guid.NewGuid().GetHashCode());
				int value2 = random.Next() % 2;

				int xor = XOR(value1, value2);

				slp.Train(new List<double>() { value1, value2 }, new List<double>() { xor });
				//if (percent == (int)percent)
				//{
				//	Console.Clear();
				//	Console.WriteLine(percent);
				//}
			}
			Console.WriteLine(n / nTestCase);
			Console.Read();
		}

		private static SingleLayerPerceptron SingleLayerPerceptronXORProgram(int nInput, int nOutput, int neuronsPerLayers)
		{
			SingleLayerPerceptron slp = new SingleLayerPerceptron(nInput, neuronsPerLayers, nOutput);
			for (int i = 0; i < 100000; i++)
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				int value1 = random.Next() % 2;
				random = new Random(Guid.NewGuid().GetHashCode());
				int value2 = random.Next() % 2;

				int xor = XOR(value1, value2);

				slp.Train(new List<double>() { value1, value2 }, new List<double>() { xor });
			}
			double n = 0;
			for (int i = 0; i < 2000; i++)
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				int value1 = random.Next() % 2;
				random = new Random(Guid.NewGuid().GetHashCode());
				int value2 = random.Next() % 2;

				int xor = XOR(value1, value2);

				var guess = slp.Predict(new List<double>() { value1, value2 });
				double result = guess[0] > 0.5 ? 1 : 0;
				bool correct = result == xor;
				Console.WriteLine("xor : " + xor);
				Console.WriteLine("guess : " + guess[0]);
				Console.WriteLine("--");
				if (correct)
				{
					n++;
				}
			}
			Console.WriteLine(n / 2000);
			Console.Read();
			return slp;
		}

		private static void ProgramFormula(int nInput, int nOutput, int neuronsPerLayers)
		{
			SingleLayerPerceptron slp = new SingleLayerPerceptron(nInput, neuronsPerLayers, nOutput);
			for (int i = 0; i < 1000000; i++)
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				double value1 = random.NextDouble();
				random = new Random(Guid.NewGuid().GetHashCode());
				double value2 = random.NextDouble();

				double y = Formula(value1);

				double result = value2 > y ? 1 : 0;

				slp.Train(new List<double>() { value1, value2 }, new List<double> { result });
			}

			double n = 0;
			for (int i = 0; i < 200; i++)
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				double value1 = random.NextDouble();
				random = new Random(Guid.NewGuid().GetHashCode());
				double value2 = random.NextDouble();

				double y = Formula(value1);

				double result = value2 > y ? 1 : 0;
				var guess = slp.Predict(new List<double>() { value1, value2 });
				double proj = guess[0] > 0.5 ? 1 : 0;

				bool correct = proj == result;
				Console.WriteLine("result : " + result);
				Console.WriteLine("guess : " + guess[0]);
				Console.WriteLine("correct : " + correct);
				Console.WriteLine("--");

				if (correct)
				{
					n++;
				}
			}
			Console.WriteLine(n / 200);
		}

		private static void SinglePerceptromProgram2(int nInput)
		{
			SinglePerceptron nn2 = new SinglePerceptron(nInput);
			for (int i = 0; i < 1000; i++)
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				double value1 = random.NextDouble();
				random = new Random(Guid.NewGuid().GetHashCode());
				double value2 = random.NextDouble();

				double y = Formula(value1);

				double result = value2 > y ? 1 : -1;

				nn2.Train(new List<double>() { value1, value2 }, result);
			}

			double n = 0;
			for (int i = 0; i < 200; i++)
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				double value1 = random.NextDouble();
				random = new Random(Guid.NewGuid().GetHashCode());
				double value2 = random.NextDouble();

				double y = Formula(value1);

				double result = value2 > y ? 1 : -1;
				var guess = nn2.Predict(new List<double>() { value1, value2 }).First();
				bool correct = result / guess > 0;
				Console.WriteLine("result : " + result);
				Console.WriteLine("guess : " + guess);
				Console.WriteLine("correct : " + correct);
				Console.WriteLine("--");

				if (correct)
				{
					n++;
				}
			}
			Console.WriteLine(n / 200);
		}

		private static double Formula(double value1)
		{
			return Math.Sin(value1 * 10);
		}

		private static double Formula(double value1, double value2, double value3)
		{
			return 0.8 * value1 + 0.1 * value2 + 0.5 * value3;
		}

		private static void SinglePersptronProgram(int nInput)
		{
			SinglePerceptron nn2 = new SinglePerceptron(nInput);
			List<double> data = new List<double>() { 0.2, 0.6, 0.5 };
			for (int i = 0; i < 1000; i++)
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				int value1 = random.Next() % 2;
				random = new Random(Guid.NewGuid().GetHashCode());
				int value2 = random.Next() % 2;

				int xor = AndBool(value1, value2);
				if (xor == 0)
				{
					xor = -1;
				}

				nn2.Train(new List<double>() { value1, value2 }, xor);
			}

			for (int i = 0; i < 200; i++)
			{
				Random random = new Random(Guid.NewGuid().GetHashCode());
				int value1 = random.Next() % 2;
				random = new Random(Guid.NewGuid().GetHashCode());
				int value2 = random.Next() % 2;

				int xor = AndBool(value1, value2);

				double guess = nn2.Predict(new List<double>() { value1, value2 }).First();
				Console.WriteLine("xor : " + xor);
				Console.WriteLine("guess : " + guess);
				Console.WriteLine("correct : " + (xor / guess > 0));
				Console.WriteLine("--");
			}
		}

		private static int AndBool(int value1, int value2)
		{
			return value1 == 1 && value2 == 1 ? 1 : 0;
		}
		private static int XOR(int value1, int value2)
		{
			return value1 ^ value2;
		}

		private static void RenderMatrix(List<List<double>> matrix)
		{
			foreach (List<double> row in matrix)
			{
				foreach (double value in row)
					Console.Write(string.Concat(value, " "));
				Console.WriteLine();
			}
			Console.WriteLine();
		}
	}
}