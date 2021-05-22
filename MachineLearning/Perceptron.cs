using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
	/// <summary>
	/// Neuronal Network with only one neuron
	/// </summary>
	public class SinglePerceptron : IMachineLearning
	{
		#region Constants
		const double LEARNING_RATE = 0.1;
		#endregion

		#region Public Methods
		public int NumberInput;
		#endregion

		#region Private Properties
		List<double> weights;
		double bias;
		#endregion

		#region Public Methods
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
					weights[i] += LEARNING_RATE * (result - guess) * datas[i];
				}

				bias += LEARNING_RATE * (result - guess);
			}

		}

		public void Train(List<double> datas, List<double> expectedValues)
		{
			if (expectedValues.Count > 0)
				Train(datas, expectedValues.First());
		}

		public void Train(List<List<double>> dataSet, List<List<double>> expectedValues)
		{
			for (int i = 0; i < dataSet.Count; i++)
			{
				Train(dataSet[i], expectedValues[i]);
			}
		}

		List<double> IMachineLearning.Predict(List<double> datas)
		{
			return new List<double> { Predict(datas) };
		}
		#endregion

		#region Private Methods
		private static double Sign(double number)
		{
			if (number > 0)
				return 1;

			return -1;
		}
		#endregion
	}
}