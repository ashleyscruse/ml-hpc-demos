# Import libraries

# Generate large synthetic datasets and perform matrix operations
import numpy as np  

# ML algorithm for timing demonstrations
from sklearn.linear_model import LinearRegression  

# Measure training time to show performance degradation
import time  

# Monitor real-time memory usage during demos
import psutil  

# Get system information like CPU count
import os

def demo_scaling(sizes):
   """
   Demonstrate how training time scales with dataset size for machine learning models.
   
   Args:
       sizes (int or list): Single dataset size (int) or list of dataset sizes to test.
                          Each size represents the number of training samples.
                          Example: 1000 or [1000, 10000, 100000]
   
   How it works:
       The function takes the arguments and generates synthetic data with 10 features/columns. 
       Then the function runs LinearRegression from sklearn.linear_model and times the training.
       
   Returns:
       tuple: (sizes_list, times_list)
              - sizes_list: The dataset sizes that were tested  
              - times_list: Training times in seconds for each dataset size
   
   Usage:
       # Test single dataset size
       sizes, times = demo_scaling(10000)
       
       # Test multiple dataset sizes  
       sizes, times = demo_scaling([1000, 10000, 100000])
       
       # Get help
       help(demo_scaling)
   """
   # Handle single size input
   if isinstance(sizes, int):
       sizes = [sizes]
   
   print("=== Training Time Scaling ===")

    # create the list variable to store times
   times = []
   
   for size in sizes:
       #print(f"Training on {size:,} samples \n")
       
       # Generate synthetic data with n size rows and 10 columns
       X = np.random.randn(size, 10)  # 10 features
       y = np.random.randn(size)
       
       # Time the training on LinearRegression from sklearn.linear_model 
       start = time.time()
       model = LinearRegression()
       model.fit(X, y)
       end = time.time()

       # Calulate the time it takes to train the model
       training_time = end - start

       # Add the time information to the time list variable 
       times.append(training_time)
       
       # Prints results for user review
       print(f"  Dataset size: {size:,} samples, Training time: {training_time:.4f}s")
       print(times)
   # Returns the results 
   return sizes, times

def show_memory_usage():
   """
   Show current memory usage of the Python process.
   
   Returns:
       float: Memory usage in megabytes (MB)
       
   Usage:
       current_mem = show_memory_usage()
       print(f"Using {current_mem:.2f} MB of RAM")
   """
   # Get the current Python process information
   process = psutil.Process(os.getpid())
   
   # Get memory info and convert from bytes to megabytes
   memory_mb = process.memory_info().rss / 1024 / 1024
   return memory_mb

def demo_memory_explosion(array_sizes=None):
   """
   Demonstrate memory usage with increasingly large arrays.
   
   Args:
       array_sizes (list, optional): List of tuples (rows, cols, label) for array sizes.
                                   Default: [(1000, 1000, "1K x 1K"), 
                                            (5000, 5000, "5K x 5K"), 
                                            (10000, 10000, "10K x 10K")]
   
   Returns:
       tuple: (memory_usage_list, labels_list) 
              - memory_usage_list: List of memory usage at each step
              - labels_list: List of labels for each measurement
   
   Usage:
       # Use default sizes
       memory_usage, labels = demo_memory_explosion()
       
       # Use custom sizes
       custom_sizes = [(2000, 2000, "2K x 2K"), (8000, 8000, "8K x 8K")]
       memory_usage, labels = demo_memory_explosion(custom_sizes)
   """
   print("Memory Requirements")
   
   # Record initial memory usage before creating any arrays
   print(f"Initial memory usage: {show_memory_usage():.2f} MB")
   
   # Set default array sizes if none provided
   if array_sizes is None:
       array_sizes = [
           (1000, 1000, "1K x 1K"),
           (5000, 5000, "5K x 5K"),
           (10000, 10000, "10K x 10K")
       ]
   
   # Initialize storage for arrays and tracking variables
   arrays = []
   memory_usage = [show_memory_usage()]
   labels = ["Initial"]
   
   # Loop through each array size and create arrays
   for rows, cols, label in array_sizes:
       try:
           # Create array with random numbers
           arr = np.random.randn(rows, cols)
           
           # Store array to prevent garbage collection
           arrays.append(arr)
           
           # Measure memory usage after creating array
           current_memory = show_memory_usage()
           memory_usage.append(current_memory)
           labels.append(label)
           
           # Calculate and display array size information
           array_size_mb = arr.nbytes / (1024 * 1024)
           print(f"Created {label} array ({array_size_mb:.1f} MB)")
           print(f"  Current memory usage: {current_memory:.2f} MB")
           
       # Handle case where system runs out of memory
       except MemoryError:
           print(f"Out of memory when creating {label} array!")
           break
   
   # Return results for plotting or further analysis
   return memory_usage, labels

def demo_parameter_explosion(architectures=None):
   """
   Show how neural network parameters explode with network size.
   
   Args:
       architectures (list, optional): List of tuples (name, layer_sizes) for network architectures.
                                     Each layer_sizes is a list of integers representing neurons per layer.
                                     Default: [("Simple", [784, 10, 10]),
                                              ("Medium", [784, 128, 64, 10]),
                                              ("Large", [784, 512, 512, 10]),
                                              ("Very Large", [784, 1024, 1024, 1024, 10])]
   
   Returns:
       tuple: (param_counts_list, arch_names_list)
              - param_counts_list: List of total parameters for each architecture
              - arch_names_list: List of architecture names
   
   Usage:
       # Use default architectures
       param_counts, names = demo_parameter_explosion()
       
       # Use custom architectures
       custom_archs = [("Tiny", [784, 32, 10]), ("Huge", [784, 2048, 2048, 10])]
       param_counts, names = demo_parameter_explosion(custom_archs)
       
       # Get help
       help(demo_parameter_explosion)
   """
   print("Demo 3: Deep Learning Parameter Count")
   
   # Set default network architectures if none provided
   if architectures is None:
       architectures = [
           ("Simple", [784, 10, 10]),
           ("Medium", [784, 128, 64, 10]),
           ("Large", [784, 512, 512, 10]),
           ("Very Large", [784, 1024, 1024, 1024, 10])
       ]
   
   # Initialize lists to store results
   param_counts = []
   arch_names = []
   
   # Loop through each network architecture
   for name, layers in architectures:
       # Initialize parameter counter for this network
       total_params = 0
       
       # Calculate parameters for each layer connection
       for i in range(len(layers) - 1):
           # Weight matrix: connects layer i to layer i+1
           # Size = input_neurons × output_neurons
           weight_params = layers[i] * layers[i+1]
           
           # Bias vector: one bias per output neuron
           bias_params = layers[i+1]
           
           # Total parameters for this layer
           layer_params = weight_params + bias_params
           total_params += layer_params
       
       # Store results for plotting
       param_counts.append(total_params)
       arch_names.append(name)
       
       # Display network information
       print(f"{name} network:")
       print(f"  Architecture: {' → '.join(map(str, layers))}")
       print(f"  Total parameters: {total_params:,}")
       
       # Calculate memory requirements (4 bytes per float32 parameter)
       memory_mb = (total_params * 4) / (1024 * 1024)
       print(f"  Memory for weights: {memory_mb:.2f} MB")
       print()
   
   # Return results for plotting or further analysis
   return param_counts, arch_names

def show_real_world_examples(examples=None):
   """
   Display real-world ML/HPC examples showing computational requirements.
   
   Args:
       examples (list, optional): List of dictionaries containing model information.
                                Each dictionary should have keys: "Model", "Parameters", 
                                "Training Time", "Dataset Size"
                                Default: Includes ResNet-50, GPT-3, BERT-Large, and Genome examples
   
   Returns:
       list: The examples that were displayed (useful for further processing)
   
   Usage:
       # Use default examples
       show_real_world_examples()
       
       # Use custom examples
       custom_examples = [
           {
               "Model": "Custom CNN",
               "Parameters": "10M",
               "Training Time": "24 GPU-hours", 
               "Dataset Size": "100K images"
           }
       ]
       show_real_world_examples(custom_examples)
       
       # Get help
       help(show_real_world_examples)
   """
   print("=== Real-world Examples ===")
   
   # Set default examples if none provided
   if examples is None:
       examples = [
           {
               "Model": "ResNet-50 (ImageNet)",
               "Parameters": "25.6M",
               "Training Time": "256 GPU-hours",
               "Dataset Size": "1.2M images"
           },
           {
               "Model": "GPT-3",
               "Parameters": "175B",
               "Training Time": "Several thousand GPU-months",
               "Dataset Size": "45TB of text"
           },
           {
               "Model": "BERT-Large",
               "Parameters": "340M",
               "Training Time": "64 TPU-hours",
               "Dataset Size": "16GB of text"
           },
           {
               "Model": "Human Genome Alignment",
               "Parameters": "3B base pairs",
               "Training Time": "Hours to days on HPC",
               "Dataset Size": "200GB per genome"
           }
       ]
   
   # Display each example with formatting
   for example in examples:
       # Print model name
       print(f"{example['Model']}")
       
       # Print model statistics with indentation
       print(f"   Parameters: {example['Parameters']}")
       print(f"   Training Time: {example['Training Time']}")
       print(f"   Dataset Size: {example['Dataset Size']}")
       print()  # Add blank line for readability
   
   # Educational calculation section
   print("Memory Calculation Example:")
   print("GPT-3 with 175B parameters:")
   
   # Break down memory calculation step by step
   print("• Each parameter = 4 bytes (float32)")
   print("• Total memory = 175B × 4 bytes = 700GB")
   print("• This is just for model weights, not including:")
   
   # List additional memory requirements with indentation
   print("  - Gradients (another 700GB)")
   print("  - Optimizer states (another 700GB+)")
   print("  - Activation memory during training")
   
   # Show final dramatic number
   print("• Total training memory: 2-3TB+ per model replica!")
   
   # Return examples for potential further use
   return examples

def demo_memory_limits(max_sizes=None):
   """
   Safely demonstrate memory limitations by attempting to create large arrays.
   
   Args:
       max_sizes (list, optional): List of array sizes to attempt creating.
                                  Each size creates a size×size array.
                                  Default: [30000, 40000, 50000, 60000]
                                  
   Returns:
       dict: Results showing which sizes succeeded/failed and memory usage
   
   Usage:
       # Use default sizes
       demo_memory_limits()
       
       # Use custom sizes (be careful with large numbers)
       custom_sizes = [10000, 25000, 35000]
       results = demo_memory_limits(custom_sizes)
       
       # Conservative test for weaker machines
       small_test = [5000, 15000, 25000]
       results = demo_memory_limits(small_test)
       
       # Get help
       help(demo_memory_limits)
   """
   print("Demo 5: What Happens When We Hit Limits")
   
   # Set default array sizes if none provided
   if max_sizes is None:
       max_sizes = [30000, 40000, 50000, 60000]
   
   print("Let's see what happens when we try to create very large arrays...")
   print()
   
   # Initialize results tracking
   results = {
       'successful_sizes': [],
       'failed_sizes': [],
       'memory_usage': []
   }
   
   # Test each array size
   for max_size in max_sizes:
       try:
           # Show current memory before attempt
           initial_memory = show_memory_usage()
           print(f"Attempting to create {max_size:,}×{max_size:,} array...")
           print(f"  Current memory usage: {initial_memory:.2f} MB")
           
           # Attempt to create large array
           huge_array = np.random.randn(max_size, max_size)
           
           # Calculate array size in different units
           size_bytes = huge_array.nbytes
           size_mb = size_bytes / (1024**2)
           size_gb = size_bytes / (1024**3)
           
           # Show success information
           print(f"  Success! Created {size_gb:.2f}GB array ({size_mb:.1f} MB)")
           
           # Track memory usage after creation
           final_memory = show_memory_usage()
           memory_increase = final_memory - initial_memory
           print(f"  Memory increase: {memory_increase:.2f} MB")
           
           # Store successful result
           results['successful_sizes'].append(max_size)
           results['memory_usage'].append(size_mb)
           
           # Clean up immediately to free memory
           del huge_array
           print(f"  Array deleted to free memory")
           print()
           
       # Handle out of memory error
       except MemoryError:
           print(f"  Out of memory when creating {max_size:,}×{max_size:,} array!")
           print(f"  This is why we need HPC for large-scale ML.")
           results['failed_sizes'].append(max_size)
           print()
           break  # Stop trying larger sizes
           
       # Handle other potential errors
       except Exception as e:
           print(f"  Error: {e}")
           results['failed_sizes'].append(max_size)
           print()
   
   # Educational summary section
   print("This is exactly why we need:")
   print("• More RAM (up to TBs on HPC systems)")
   print("• GPU memory (up to 80GB per GPU)")
   print("• Distributed computing (split across multiple machines)")
   print("• Specialized hardware (TPUs, etc.)")
   print()
   
   # Show summary of results
   if results['successful_sizes']:
       print("Summary:")
       print(f"  Largest successful array: {max(results['successful_sizes']):,}×{max(results['successful_sizes']):,}")
       max_memory = max(results['memory_usage']) if results['memory_usage'] else 0
       print(f"  Maximum memory used: {max_memory:.1f} MB")
   
   if results['failed_sizes']:
       print(f"  Failed at size: {min(results['failed_sizes']):,}×{min(results['failed_sizes']):,}")
   
   # Return results for further analysis
   return results