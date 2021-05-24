using MachineLearning.Calculus;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
	[JsonObject]
	public class DeepLearning : BaseLearning, IMachineLearning
	{
		#region Constants
		const double LEARNING_RATE = 0.1;
		#endregion

		#region Public Properties
		public int NumberInput;
		public int NumberNeurons;
		public int NumberOutput;
		public int NumberOfLayers;
		#endregion

		#region Private Properties
		[JsonProperty]
		private List<Matrix<double>> WeightsHidden;
		[JsonProperty]
		private List<Matrix<double>> BiasHidden;
		[JsonProperty]
		private List<Matrix<double>> ActivationsHiddenSigmoid;
		#endregion

		#region Ctor
		public DeepLearning()
		{

		}

		public DeepLearning(int numberInput, int nNeurons, int numberOutput, int numberOfLayers)
		{
			NumberInput = numberInput;
			NumberNeurons = nNeurons;
			NumberOutput = numberOutput;
			NumberOfLayers = numberOfLayers;

			WeightsHidden = new List<Matrix<double>>();
			BiasHidden = new List<Matrix<double>>();
			ActivationsHiddenSigmoid = new List<Matrix<double>>();

			WeightsHidden.Add(CreateMatrix.Random<double>(NumberNeurons, NumberInput, new ContinuousUniform(0, 0.1)));
			BiasHidden.Add(CreateMatrix.Random<double>(NumberNeurons, 1, new ContinuousUniform(0, 0.1)));

			//Set the hidden weights with random values
			for (int k = 0; k < NumberOfLayers - 1; k++)
			{
				WeightsHidden.Add(CreateMatrix.Random<double>(NumberNeurons, NumberNeurons, new ContinuousUniform(0, 0.1)));
				BiasHidden.Add(CreateMatrix.Random<double>(NumberNeurons, 1, new ContinuousUniform(0, 0.1)));
			}

			WeightsHidden.Add(CreateMatrix.Random<double>(NumberOutput, NumberNeurons, new ContinuousUniform(0, 0.1)));
			BiasHidden.Add(CreateMatrix.Random<double>(NumberOutput, 1, new ContinuousUniform(0, 0.1)));
		}

		public void Save(string path)
		{
			File.WriteAllText(path, JsonConvert.SerializeObject(this));
		}
		#endregion

		#region Public Methods
		/// <summary>
		/// Seems to Work
		/// </summary>
		/// <param name="data"></param>
		/// <returns></returns>
		internal override Matrix<double> Predict(IEnumerable<IEnumerable<double>> data)
		{
			ActivationsHiddenSigmoid.Clear();
			var datasMatrix = CreateMatrix.DenseOfRows(data);

			var actHidden = (WeightsHidden.First() * datasMatrix) + BiasHidden.First();
			ActivationsHiddenSigmoid.Add(actHidden.Map(Tools.Sigmoid));

			for (int i = 1; i < NumberOfLayers + 1; i++)
			{
				ActivationsHiddenSigmoid.Add(((WeightsHidden[i] * ActivationsHiddenSigmoid[i - 1]) + BiasHidden[i]).Map(Tools.Sigmoid));
			}

			return ActivationsHiddenSigmoid.Last();
		}

		public override void Train(IList<double> datas, IList<double> expectedValues)
		{
			var errors = new List<Matrix<double>>();

			var deltaWeight = new List<Matrix<double>>();
			var deltaBiais = new List<Matrix<double>>();

			IEnumerable<IEnumerable<double>> resultMatrix = expectedValues.ToMatrix();
			IEnumerable<IEnumerable<double>> inputMatrix = datas.ToMatrix();

			var result = CreateMatrix.DenseOfRows(resultMatrix);
			var input = CreateMatrix.DenseOfRows(inputMatrix);
			var guess = Predict(inputMatrix);

			var errorM = result - guess;
			errors.Add(errorM);

			deltaWeight.Add(errorM.GetDeltaError(guess, LEARNING_RATE) * ActivationsHiddenSigmoid[NumberOfLayers - 1].Transpose());
			deltaBiais.Add(errorM.GetDeltaError(guess, LEARNING_RATE));

			for (int i = 1; i < NumberOfLayers; i++)
			{
				//weightsHidden is backpropagated but Errors and deltaWeights not !!!
				var errorHiddenMatrix = WeightsHidden[NumberOfLayers + 1 - i].Transpose() * errors[i - 1];
				errors.Add(errorHiddenMatrix);

				deltaWeight.Add(errorHiddenMatrix.GetDeltaError(ActivationsHiddenSigmoid[NumberOfLayers - i], LEARNING_RATE) * ActivationsHiddenSigmoid[NumberOfLayers - i - 1].Transpose());
				deltaBiais.Add(errorHiddenMatrix.GetDeltaError(ActivationsHiddenSigmoid[NumberOfLayers - i - 1], LEARNING_RATE));
			}

			var errorFirstMatrix = WeightsHidden[1].Transpose() * errors[NumberOfLayers - 1];
			deltaWeight.Add(errorFirstMatrix.GetDeltaError(ActivationsHiddenSigmoid[0], LEARNING_RATE) * input.Transpose());
			deltaBiais.Add(errorFirstMatrix.GetDeltaError(ActivationsHiddenSigmoid[0], LEARNING_RATE));

			for (int i = 0; i < deltaWeight.Count; i++)
			{
				WeightsHidden[i] += deltaWeight[deltaWeight.Count - 1 - i];
			}

			for (int i = 0; i < deltaBiais.Count; i++)
			{
				BiasHidden[i] = BiasHidden[i] + deltaBiais[deltaBiais.Count - 1 - i];
			}
		}
		#endregion

		#region Private Methods
		public void Load(string path)
		{
			string json = File.ReadAllText(path);
			var jsonObject = JsonConvert.DeserializeObject(json) as JObject;
			NumberInput = GetValue(jsonObject, nameof(NumberInput));
			NumberNeurons = GetValue(jsonObject, nameof(NumberNeurons));
			NumberOfLayers = GetValue(jsonObject, nameof(NumberOfLayers));
			NumberOutput = GetValue(jsonObject, nameof(NumberOutput));

			int GetValue(JObject jsonObject, string name)
			{
				return int.Parse(((JValue)jsonObject[name]).Value.ToString());
			}

			var weightsHidden = new List<Matrix<double>>();
			var biasHidden = new List<Matrix<double>>();

			ExtractMatrix(jsonObject, weightsHidden, nameof(this.WeightsHidden));
			ExtractMatrix(jsonObject, biasHidden, nameof(this.BiasHidden), true);

			void ExtractMatrix(JObject jsonObject, List<Matrix<double>> collection, string name, bool isBiais = false)
			{
				var array = jsonObject[name] as JArray;
				foreach (var obj in array.OfType<JObject>())
				{
					Matrix<double> matrix;
					if (obj == array.First())
					{
						matrix = ExtractMatrix(NumberNeurons, isBiais ? 1 : NumberInput, obj);
					}
					else if (obj == array.Last())
					{
						matrix = ExtractMatrix(NumberOutput, isBiais ? 1 : NumberInput, obj);
					}
					else
					{
						matrix = ExtractMatrix(NumberNeurons, isBiais ? 1 : NumberNeurons, obj);
					}
					collection.Add(matrix);
				}

				static Matrix<double> ExtractMatrix(int size1, int size2, JObject obj)
				{
					Matrix<double> matrix;
					var values = obj["Values"];
					var arrayOfDouble = ((JArray)values).Select(a => ((JValue)a).Value).Cast<double>().ToArray();
					double[,] multiDimArray = new double[size1, size2];
					for (int i = 0; i < size1; i++)
					{
						for (int j = 0; j < size2; j++)
						{
							multiDimArray[i, j] = arrayOfDouble[size2 * i + j];
						}
					}
					matrix = Matrix<double>.Build.DenseOfArray(multiDimArray);
					return matrix;
				}
			}
		}

		#endregion
	}
}