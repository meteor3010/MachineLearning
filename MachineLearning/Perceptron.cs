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

		public double[] Predict(IList<double> datas)
		{
			double result = 0;
			for (int i = 0; i < datas.Count; i++)
			{
				result += weights[i] * datas[i] + bias;
			}

			return new double[] { result };
		}

		public void Train(IList<double> datas, double result)
		{
			double guess = Predict(datas).First();

			if (Math.Sign(guess) != result)
			{
				for (int i = 0; i < weights.Count; i++)
				{
					weights[i] += LEARNING_RATE * (result - guess) * datas[i];
				}

				bias += LEARNING_RATE * (result - guess);
			}
		}

		public void Train(IList<double> datas, IList<double> expectedValues)
		{
			if (expectedValues.Count > 0)
				Train(datas, expectedValues.First());
		}

		#endregion

		#region Private Methods
		#endregion
	}
}