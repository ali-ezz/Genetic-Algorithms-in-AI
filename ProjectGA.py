import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
from datetime import datetime
import os
import traceback
import threading
import random
import itertools
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ------------------ Genetic Algorithm ------------------

class GeneticAlgorithm:
    def __init__(self, fitness_function, bounds, dimensions, 
                 population_size=100, crossover_rate=0.8, mutation_rate=0.1, 
                 generations=1000):
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.dimensions = dimensions
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        # Tracking variables
        self.best_solutions = []
        self.best_fitness_values = []
        self.avg_fitness_values = []
        self.diversity_values = []
        self.mutation_rates = []
        self.execution_time = 0
        
    def initialize_population(self):
        """Initialize a random population within the bounds"""
        lower_bound, upper_bound = self.bounds
        return lower_bound + np.random.rand(self.population_size, self.dimensions) * (upper_bound - lower_bound)
    
    def evaluate_fitness(self, population):
        """Evaluate fitness for each individual"""
        return np.array([self.fitness_function(individual) for individual in population])
    
    def selection(self, population, fitness):
        """Tournament selection"""
        selected = np.zeros_like(population)
        for i in range(len(population)):
            # Randomly select 3 individuals
            tournament_indices = np.random.choice(len(population), 3, replace=False)
            tournament_fitness = fitness[tournament_indices]
            winner = tournament_indices[np.argmin(tournament_fitness)]
            selected[i] = population[winner]
        return selected
    
    def crossover(self, parents):
        """Uniform crossover"""
        offspring = np.copy(parents)
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents) and np.random.random() < self.crossover_rate:
                # Create a mask for crossover
                mask = np.random.random(self.dimensions) < 0.5
                # Swap values where mask is True
                offspring[i, mask] = parents[i+1, mask]
                offspring[i+1, mask] = parents[i, mask]
        return offspring
    
    def mutation(self, offspring):
        """Mutation with adaptive rate"""
        lower_bound, upper_bound = self.bounds
        mutation_mask = np.random.random(offspring.shape) < self.mutation_rate
        mutation_values = lower_bound + np.random.random(offspring.shape) * (upper_bound - lower_bound)
        offspring[mutation_mask] = mutation_values[mutation_mask]
        return offspring
    
    def calculate_diversity(self, population):
        """Calculate population diversity"""
        return np.mean(np.std(population, axis=0))
    
    def run(self):
        """Main optimization loop"""
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness = self.evaluate_fitness(population)
            
            # Track best solution
            best_idx = np.argmin(fitness)
            self.best_solutions.append(population[best_idx])
            self.best_fitness_values.append(fitness[best_idx])
            self.avg_fitness_values.append(np.mean(fitness))
            
            # Calculate diversity
            diversity = self.calculate_diversity(population)
            self.diversity_values.append(diversity)
            self.mutation_rates.append(self.mutation_rate)
            
            # Selection
            parents = self.selection(population, fitness)
            
            # Crossover
            offspring = self.crossover(parents)
            
            # Mutation
            population = self.mutation(offspring)
        
        # Final evaluation
        fitness = self.evaluate_fitness(population)
        best_idx = np.argmin(fitness)
        
        # Record execution time
        self.execution_time = time.time() - start_time
        
        return self.best_solutions[-1], self.best_fitness_values[-1]

# ------------------ Differential Evolution ------------------

class DifferentialEvolution:
    def __init__(self, fitness_function, bounds, dimensions, 
                 population_size=100, crossover_rate=0.8, differential_weight=0.8, generations=1000):
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.dimensions = dimensions
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.differential_weight = differential_weight
        self.generations = generations
        
        # Tracking variables
        self.best_solutions = []
        self.best_fitness_values = []
        self.avg_fitness_values = []
        self.execution_time = 0
        
    def initialize_population(self):
        lower_bound, upper_bound = self.bounds
        return lower_bound + np.random.rand(self.population_size, self.dimensions) * (upper_bound - lower_bound)
    
    def run(self):
        start_time = time.time()
        population = self.initialize_population()
        fitness = np.array([self.fitness_function(ind) for ind in population])
        best_idx = np.argmin(fitness)
        
        for gen in range(self.generations):
            for i in range(self.population_size):
                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.differential_weight * (b - c), self.bounds[0], self.bounds[1])
                # Crossover
                cross_points = np.random.rand(self.dimensions) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimensions)] = True
                trial = np.where(cross_points, mutant, population[i])
                f = self.fitness_function(trial)
                if f < fitness[i]:
                    population[i] = trial
                    fitness[i] = f
            best_idx = np.argmin(fitness)
            self.best_solutions.append(population[best_idx])
            self.best_fitness_values.append(fitness[best_idx])
            self.avg_fitness_values.append(np.mean(fitness))
        self.execution_time = time.time() - start_time

        return self.best_solutions[-1], self.best_fitness_values[-1]

# ------------------ GUI ------------------

class MultiAlgorithmGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Window Configuration
        self.title("Metaheuristic Optimizer - Multi Algorithm (GA & DE)")
        self.geometry("1300x930")
        self.configure(bg='#f0f0f0')
        
        # Setup Variables
        self._setup_variables()
        
        # Create Main Layout
        self._create_main_layout()
        
        # Setup Error Handling
        self._setup_error_handling()
    
    def _setup_variables(self):
        # Optimization Functions
        self.optimization_functions = {
            "Booth Function": self._booth_function,
            "Matyas Function": self._matyas_function,
            "Rosenbrock Function": self._rosenbrock_function,
            "Sphere Function": self._sphere_function,
            "Zakharov Function": self._zakharov_function
        }
        self.function_var = tk.StringVar(value="Sphere Function")
        # Algorithm selection (now as a multi-selection)
        self.algorithm_var = tk.StringVar(value="Both")
        self.algorithms_list = ["Genetic Algorithm", "Differential Evolution", "Both"]
        
        # Control variables
        self.population_size_var = tk.IntVar(value=100)
        self.max_generations_var = tk.IntVar(value=500)
        self.crossover_rate_var = tk.DoubleVar(value=0.8)
        self.mutation_rate_var = tk.DoubleVar(value=0.1)  # Only for GA
        self.differential_weight_var = tk.DoubleVar(value=0.8)  # Only for DE
        
        # State Variables
        self.is_running = False
        self.optimization_results = {}  # store per algorithm
    
    def _create_main_layout(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self._create_setup_tab()
        self._create_results_tab()
        self._create_visualization_tab()
        self._create_analysis_tab()
    
    def _create_setup_tab(self):
        setup_frame = ttk.Frame(self.notebook)
        self.notebook.add(setup_frame, text="Optimization Setup")
        
        # Algorithm Selection
        ttk.Label(setup_frame, text="Algorithm(s):").pack(pady=(10,5))
        algorithm_dropdown = ttk.Combobox(
            setup_frame,
            textvariable=self.algorithm_var,
            values=self.algorithms_list,
            state="readonly",
            width=30
        )
        algorithm_dropdown.pack(pady=5)
        
        # Function Selection
        ttk.Label(setup_frame, text="Objective Function:").pack(pady=(15,5))
        function_dropdown = ttk.Combobox(
            setup_frame, 
            textvariable=self.function_var, 
            values=list(self.optimization_functions.keys()),
            state="readonly",
            width=30
        )
        function_dropdown.pack(pady=5)
        
        # Parameter Inputs
        params = [
            ("Population Size:", self.population_size_var),
            ("Max Generations:", self.max_generations_var),
            ("Crossover Rate:", self.crossover_rate_var),
            ("Mutation Rate (GA only):", self.mutation_rate_var),
            ("Differential Weight F (DE only):", self.differential_weight_var)
        ]
        
        for label, var in params:
            frame = ttk.Frame(setup_frame)
            frame.pack(fill='x', padx=50, pady=5)
            ttk.Label(frame, text=label).pack(side=tk.LEFT)
            entry = ttk.Entry(frame, textvariable=var, width=20)
            entry.pack(side=tk.RIGHT)
        
        # Start Optimization Button
        start_btn = ttk.Button(
            setup_frame, 
            text="Start Optimization", 
            command=self._validate_and_start_optimization
        )
        start_btn.pack(pady=20)
    
    def _validate_and_start_optimization(self):
        try:
            # Input Validation
            population_size = self.population_size_var.get()
            max_generations = self.max_generations_var.get()
            crossover_rate = self.crossover_rate_var.get()
            mutation_rate = self.mutation_rate_var.get()
            differential_weight = self.differential_weight_var.get()
            
            if population_size <= 0:
                raise ValueError("Population size must be positive")
            if max_generations <= 0:
                raise ValueError("Generations must be positive")
            if not (0 <= crossover_rate <= 1):
                raise ValueError("Crossover rate must be between 0 and 1")
            if not (0 <= mutation_rate <= 1):
                raise ValueError("Mutation rate must be between 0 and 1")
            if not (0 <= differential_weight <= 2):
                raise ValueError("Differential weight must be between 0 and 2")
            
            threading.Thread(
                target=self._run_optimization, 
                daemon=True
            ).start()
        
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            self._handle_unexpected_error(e)
    
    def _run_optimization(self):
        try:
            self.is_running = True
            function = self.optimization_functions[self.function_var.get()]
            # Determine dimensions & bounds
            dimensions_map = {
                "Booth Function": 2,
                "Matyas Function": 2,
                "Rosenbrock Function": 4,
                "Sphere Function": 5,
                "Zakharov Function": 5
            }
            dimensions = dimensions_map[self.function_var.get()]
            bounds_map = {
                "Booth Function": (-10, 10),
                "Matyas Function": (-10, 10),
                "Rosenbrock Function": (-5, 10),
                "Sphere Function": (-5.12, 5.12),
                "Zakharov Function": (-10, 10)
            }
            bounds = bounds_map[self.function_var.get()]
            pop_size = self.population_size_var.get()
            generations = self.max_generations_var.get()
            crossover_rate = self.crossover_rate_var.get()
            mutation_rate = self.mutation_rate_var.get()
            differential_weight = self.differential_weight_var.get()
            
            algo_choice = self.algorithm_var.get()
            results = {}
            
            # Always run on a fresh random seed for fairness
            seed = int(time.time())
            np.random.seed(seed)
            random.seed(seed)
            
            # Run GA if selected
            if algo_choice in ["Genetic Algorithm", "Both"]:
                ga = GeneticAlgorithm(
                    fitness_function=function,
                    bounds=bounds,
                    dimensions=dimensions,
                    population_size=pop_size,
                    generations=generations,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate
                )
                ga_best_sol, ga_best_fit = ga.run()
                results['Genetic Algorithm'] = {
                    'best_solution': ga_best_sol.tolist(),
                    'best_fitness': float(ga_best_fit),
                    'full_results': ga,
                    'execution_time': ga.execution_time
                }
            
            # DE
            if algo_choice in ["Differential Evolution", "Both"]:
                # Ensure same seed for fair comparison
                np.random.seed(seed)
                random.seed(seed)
                de = DifferentialEvolution(
                    fitness_function=function,
                    bounds=bounds,
                    dimensions=dimensions,
                    population_size=pop_size,
                    generations=generations,
                    crossover_rate=crossover_rate,
                    differential_weight=differential_weight
                )
                de_best_sol, de_best_fit = de.run()
                results['Differential Evolution'] = {
                    'best_solution': de_best_sol.tolist(),
                    'best_fitness': float(de_best_fit),
                    'full_results': de,
                    'execution_time': de.execution_time
                }
            
            self.optimization_results = results
            self.after(0, self._optimization_complete)
        
        except Exception as e:
            self.after(0, lambda: self._handle_optimization_error(e))
        finally:
            self.is_running = False
    
    def _optimization_complete(self):
        self._update_results_tab()
        self._update_visualization_tab()
        self._update_analysis_tab()
        msg = ""
        for alg, result in self.optimization_results.items():
            msg += f"{alg} - Best Fitness: {result['best_fitness']:.6f}\n"
        messagebox.showinfo("Optimization Complete", msg)
    
    def _update_results_tab(self):
        self.results_text.delete(1.0, tk.END)
        if not self.optimization_results:
            return
        out = []
        for alg, result in self.optimization_results.items():
            out.append(f"Algorithm: {alg}\n"
                       f"Best Solution: {result['best_solution']}\n"
                       f"Best Fitness: {result['best_fitness']:.6f}\n"
                       "------------------------------\n")
        params = (
            f"Objective Function: {self.function_var.get()}\n"
            f"Population Size: {self.population_size_var.get()}\n"
            f"Max Generations: {self.max_generations_var.get()}\n"
            f"Crossover Rate: {self.crossover_rate_var.get()}\n"
            f"Mutation Rate (GA): {self.mutation_rate_var.get()}\n"
            f"Differential Weight (DE): {self.differential_weight_var.get()}\n"
        )
        self.results_text.insert(tk.END, params + "\n" + "".join(out))
    
    def _update_visualization_tab(self):
        self.fig.clear()
        colors = {'Genetic Algorithm': 'tab:blue', 'Differential Evolution': 'tab:orange'}
        # Get both results (if available)
        subplot_cnt = 2 if len(self.optimization_results) > 1 else 1
        total_rows = 2
        total_cols = 2 * subplot_cnt
        # Best Fitness vs Generation, Average Fitness vs Generation
        for i, (alg, result) in enumerate(self.optimization_results.items()):
            res = result['full_results']
            # Best fitness
            ax1 = self.fig.add_subplot(total_rows, total_cols, 1 + 2 * i)
            ax1.plot(res.best_fitness_values, color=colors.get(alg, None))
            ax1.set_title(f'{alg}: Best Fitness')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Best Fitness')
            # Average fitness
            ax2 = self.fig.add_subplot(total_rows, total_cols, 2 + 2 * i)
            ax2.plot(res.avg_fitness_values, color=colors.get(alg, None))
            ax2.set_title(f'{alg}: Average Fitness')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Average Fitness')
            # Population diversity & mutation rate (only GA)
            if alg == 'Genetic Algorithm':
                ax3 = self.fig.add_subplot(total_rows, total_cols, 2 * subplot_cnt + 1)
                ax3.plot(res.diversity_values)
                ax3.set_title('GA: Population Diversity')
                ax3.set_xlabel('Generation')
                ax3.set_ylabel('Diversity')
                ax4 = self.fig.add_subplot(total_rows, total_cols, 2 * subplot_cnt + 2)
                ax4.plot(res.mutation_rates)
                ax4.set_title('GA: Mutation Rates')
                ax4.set_xlabel('Generation')
                ax4.set_ylabel('Mutation Rate')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _update_analysis_tab(self):
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
        if not self.optimization_results:
            return
        for alg, result in self.optimization_results.items():
            res = result['full_results']
            alg_lbl = ttk.Label(self.analysis_frame, text=f"{alg}", font=('Arial', 12, "bold"))
            alg_lbl.pack(pady=(8, 4), anchor='w')
            stat_lines = [
                f"Total Generations: {len(res.best_fitness_values)}",
                f"Best Fitness: {result['best_fitness']:.6f}",
                f"Execution Time: {result['execution_time']:.2f} seconds"
            ]
            if alg == "Genetic Algorithm":
                stat_lines += [
                    f"Final Population Diversity: {getattr(res, 'diversity_values', ['?'])[-1]:.6f}",
                    f"Final Mutation Rate: {getattr(res, 'mutation_rates', ['?'])[-1]:.6f}"
                ]
            for line in stat_lines:
                label = ttk.Label(self.analysis_frame, text=line, font=('Arial', 10))
                label.pack(pady=2, anchor='w')
            sep = ttk.Separator(self.analysis_frame, orient='horizontal')
            sep.pack(fill=tk.X, pady=5)
    
    def _create_results_tab(self):
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        self.results_text = tk.Text(
            results_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=30, 
            font=('Courier', 10)
        )
        self.results_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.configure(yscrollcommand=scrollbar.set)
    
    def _create_visualization_tab(self):
        visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(visualization_frame, text="Visualization")
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=visualization_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, visualization_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_analysis_tab(self):
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
    
    def _setup_error_handling(self):
        def error_handler(exc_type, exc_value, exc_traceback):
            error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            messagebox.showerror("Unexpected Error", error_msg)
        tk.Tk.report_callback_exception = error_handler
    
    def _handle_optimization_error(self, error):
        error_msg = str(error)
        traceback_msg = traceback.format_exc()
        messagebox.showerror("Optimization Error", f"An error occurred during optimization:\n{error_msg}")
        print(traceback_msg)
    
    # Optimization test functions
    def _booth_function(self, x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    def _matyas_function(self, x):
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

    def _rosenbrock_function(self, x):
        return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

    def _sphere_function(self, x):
        return np.sum(x**2)

    def _zakharov_function(self, x):
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
        return sum1 + sum2**2 + sum2**4

def main():
    try:
        app = MultiAlgorithmGUI()
        app.mainloop()
    except Exception as e:
        print(f"Application startup error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()