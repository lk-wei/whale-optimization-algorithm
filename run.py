import argparse
import numpy as np
import csv
from src.whale_optimization import WhaleOptimization # Ensure this path is correct
np.random.seed(42)

class UrbanAgProblem:
    def __init__(self, total_land_sqm=1000, max_budget=None, max_water=None):
        self.total_land_sqm = total_land_sqm
        # Set max_budget and max_water to infinity if not provided, effectively no constraint
        self.max_budget = max_budget if max_budget is not None else float('inf')
        self.max_water = max_water if max_water is not None else float('inf')

        # Define number of crops for each category
        self.hydro_crop_count = 10
        self.container_crop_count = 10
        self.inground_crop_count = 10

        # Define bounds for each variable
        # Order: [Hydroponics %, Container %, In-Ground %, Hydroponics Crop Index, Container Crop Index, In-Ground Crop Index]
        # Percentages are (0, 100)
        # Crop indices are (0, num_crops_in_category - 1)
        self.bounds = [
            (0, 70),   # x1: Hydroponics % (Example: Max 70% to encourage diversity)
            (0, 100),  # x2: Container %
            (0, 100),  # x3: In-Ground %
            (0, self.hydro_crop_count - 1),     # c1: Hydroponics Crop Index (0 to 9)
            (0, self.container_crop_count - 1), # c2: Container Crop Index (0 to 9)
            (0, self.inground_crop_count - 1)   # c3: In-Ground Crop Index (0 to 9)
        ]
        self.num_variables = len(self.bounds)

        # --- EXPANDED CROP DATA (AT LEAST 10 PLANTS PER CATEGORY) ---
        # Hydroponics data
        self.hydro_data = {
            0: {'name': 'Hydro_Lettuce', 'yield_per_sqm': 50.0, 'water_per_sqm': 40, 'cost_per_sqm': 15},
            1: {'name': 'Hydro_Spinach', 'yield_per_sqm': 48.0, 'water_per_sqm': 38, 'cost_per_sqm': 14.5},
            2: {'name': 'Hydro_Kale', 'yield_per_sqm': 45.0, 'water_per_sqm': 42, 'cost_per_sqm': 14},
            3: {'name': 'Hydro_Basil', 'yield_per_sqm': 35.0, 'water_per_sqm': 30, 'cost_per_sqm': 12},
            4: {'name': 'Hydro_Mint', 'yield_per_sqm': 32.0, 'water_per_sqm': 28, 'cost_per_sqm': 11.5},
            5: {'name': 'Hydro_Strawberries', 'yield_per_sqm': 40.0, 'water_per_sqm': 45, 'cost_per_sqm': 16},
            6: {'name': 'Hydro_Tomatoes', 'yield_per_sqm': 38.0, 'water_per_sqm': 50, 'cost_per_sqm': 15.5},
            7: {'name': 'Hydro_Cucumbers', 'yield_per_sqm': 30.0, 'water_per_sqm': 48, 'cost_per_sqm': 13},
            8: {'name': 'Hydro_BellPeppers', 'yield_per_sqm': 28.0, 'water_per_sqm': 47, 'cost_per_sqm': 12.5},
            9: {'name': 'Hydro_Chives', 'yield_per_sqm': 20.0, 'water_per_sqm': 25, 'cost_per_sqm': 10}
        }

        # Container/Raised Beds data
        self.container_data = {
            0: {'name': 'Cont_Carrots', 'yield_per_sqm': 20.0, 'water_per_sqm': 120, 'cost_per_sqm': 7},
            1: {'name': 'Cont_Radishes', 'yield_per_sqm': 18.0, 'water_per_sqm': 110, 'cost_per_sqm': 6.5},
            2: {'name': 'Cont_BushBeans', 'yield_per_sqm': 22.0, 'water_per_sqm': 130, 'cost_per_sqm': 7.5},
            3: {'name': 'Cont_Peas', 'yield_per_sqm': 21.0, 'water_per_sqm': 125, 'cost_per_sqm': 7.2},
            4: {'name': 'Cont_Potatoes', 'yield_per_sqm': 15.0, 'water_per_sqm': 100, 'cost_per_sqm': 6},
            5: {'name': 'Cont_Onions', 'yield_per_sqm': 16.0, 'water_per_sqm': 105, 'cost_per_sqm': 6.2},
            6: {'name': 'Cont_Eggplant', 'yield_per_sqm': 25.0, 'water_per_sqm': 140, 'cost_per_sqm': 9},
            7: {'name': 'Cont_Zucchini', 'yield_per_sqm': 24.0, 'water_per_sqm': 135, 'cost_per_sqm': 8.5},
            8: {'name': 'Cont_CherryTomatoes', 'yield_per_sqm': 28.0, 'water_per_sqm': 160, 'cost_per_sqm': 10},
            9: {'name': 'Cont_Peppers', 'yield_per_sqm': 26.0, 'water_per_sqm': 155, 'cost_per_sqm': 9.5}
        }

        # In-Ground Soil-Based Farming data
        self.inground_data = {
            0: {'name': 'IG_Spinach', 'yield_per_sqm': 12.0, 'water_per_sqm': 200, 'cost_per_sqm': 4},
            1: {'name': 'IG_Cabbage', 'yield_per_sqm': 11.0, 'water_per_sqm': 190, 'cost_per_sqm': 3.8},
            2: {'name': 'IG_Broccoli', 'yield_per_sqm': 10.0, 'water_per_sqm': 180, 'cost_per_sqm': 3.5},
            3: {'name': 'IG_Corn', 'yield_per_sqm': 9.0, 'water_per_sqm': 220, 'cost_per_sqm': 4.2},
            4: {'name': 'IG_Wheat', 'yield_per_sqm': 8.0, 'water_per_sqm': 210, 'cost_per_sqm': 3.0},
            5: {'name': 'IG_Onions', 'yield_per_sqm': 13.0, 'water_per_sqm': 195, 'cost_per_sqm': 4.1},
            6: {'name': 'IG_Garlic', 'yield_per_sqm': 10.5, 'water_per_sqm': 185, 'cost_per_sqm': 3.7},
            7: {'name': 'IG_Squash', 'yield_per_sqm': 14.0, 'water_per_sqm': 215, 'cost_per_sqm': 4.5},
            8: {'name': 'IG_Pumpkins', 'yield_per_sqm': 12.5, 'water_per_sqm': 205, 'cost_per_sqm': 4.3},
            9: {'name': 'IG_Melons', 'yield_per_sqm': 11.5, 'water_per_sqm': 200, 'cost_per_sqm': 4.0}
        }

    def evaluate(self, solution):
        """
        Evaluates a given solution (combination of variables) and calculates
        total yield, total water consumption, and total cost, applying penalties
        for constraint violations.
        """
        # Ensure solution is a numpy array for consistent indexing
        solution = np.array(solution)

        x1, x2, x3 = solution[0:3] # Percentages (Hydroponics, Container, In-Ground)
        c1_idx_raw, c2_idx_raw, c3_idx_raw = solution[3:6] # Crop indices

        # --- 1. Handle Percentage Constraint and Convert to Area ---
        total_percent = x1 + x2 + x3

        # Scale down proportionally if sum exceeds 100% and is not zero
        if total_percent > 100 and total_percent != 0:
            factor = 100 / total_percent
            x1 *= factor
            x2 *= factor
            x3 *= factor
        elif total_percent <= 0: # If total percentage is zero or negative, no land is allocated
            x1, x2, x3 = 0, 0, 0

        # Ensure percentages are within bounds [0, 100] after scaling
        x1 = np.clip(x1, 0, 100)
        x2 = np.clip(x2, 0, 100)
        x3 = np.clip(x3, 0, 100)

        area_hydro = (x1 / 100) * self.total_land_sqm
        area_container = (x2 / 100) * self.total_land_sqm
        area_inground = (x3 / 100) * self.total_land_sqm

        # --- 2. Convert Categorical Crop Choices to Integers and Clamp ---
        c1_idx = int(round(c1_idx_raw))
        c2_idx = int(round(c2_idx_raw))
        c3_idx = int(round(c3_idx_raw))

        # Clamp indices to valid range
        c1_idx = np.clip(c1_idx, 0, self.hydro_crop_count - 1)
        c2_idx = np.clip(c2_idx, 0, self.container_crop_count - 1)
        c3_idx = np.clip(c3_idx, 0, self.inground_crop_count - 1)

        # --- 3. Calculate Metrics Based on Allocations and Crops ---
        total_yield = 0
        total_water = 0
        total_cost = 0

        # Hydroponics calculations
        hydro_crop_data = self.hydro_data.get(c1_idx, self.hydro_data[0])
        total_yield += area_hydro * hydro_crop_data['yield_per_sqm']
        total_water += area_hydro * hydro_crop_data['water_per_sqm']
        total_cost += area_hydro * hydro_crop_data['cost_per_sqm']

        # Container/Raised Bed calculations
        container_crop_data = self.container_data.get(c2_idx, self.container_data[0])
        total_yield += area_container * container_crop_data['yield_per_sqm']
        total_water += area_container * container_crop_data['water_per_sqm']
        total_cost += area_container * container_crop_data['cost_per_sqm']

        # In-Ground calculations
        inground_crop_data = self.inground_data.get(c3_idx, self.inground_data[0])
        total_yield += area_inground * inground_crop_data['yield_per_sqm']
        total_water += area_inground * inground_crop_data['water_per_sqm']
        total_cost += area_inground * inground_crop_data['cost_per_sqm']

        # --- 4. Apply Penalties for Exceeding Constraints ---
        penalty = 0
        if total_cost > self.max_budget:
            penalty += (total_cost - self.max_budget) * 1000 # Large penalty for exceeding budget
        if total_water > self.max_water:
            penalty += (total_water - self.max_water) * 500 # Penalty for exceeding water limit

        # --- 5. Define Fitness (Maximize Yield with Penalties) ---
        if total_yield <= 0 or (x1 + x2 + x3) <= 0: # If no yield or no land allocated, fitness is very low
            fitness = -np.inf
        else:
            fitness = total_yield - penalty

        return fitness, total_yield, total_water, total_cost # Return all metrics


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nsols", type=int, default=100, dest='nsols', help='number of solutions per generation, default: 100')
    parser.add_argument("-ngens", type=int, default=500, dest='ngens', help='number of generations, default: 500')
    parser.add_argument("-a", type=float, default=3.0, dest='a', help='woa algorithm specific parameter, controls search spread default: 2.0')
    parser.add_argument("-b", type=float, default=0.5, dest='b', help='woa algorithm specific parameter, controls spiral, default: 0.5')
    parser.add_argument("-max", default=True, dest='max', action='store_true', help='enable for maximization (default for this problem)')
    parser.add_argument("-tune", action='store_true', dest='tune_mode', help="Enable parameter tuning mode instead of single optimization run")

    args = parser.parse_args()
    return args


def tune_parameters(sof, cons):
    best_overall = (-np.inf, None)
    results = []

    for a in [1.5, 2.0, 2.5]:
        for b in [0.5, 1.0, 1.5]:
            for a_step in [0.001, 0.005]:
                print(f"\n--- Testing Params: a={a}, b={b}, a_step={a_step} ---")

                opt_alg = WhaleOptimization(
                    opt_func=sof,
                    constraints=cons,
                    nsols=30,
                    b=b,
                    a=a,
                    a_step=a_step,
                    maximize=True
                )

                opt_alg.optimize(max_iters=100)
                best_fitness, best_sol = opt_alg.print_best_solutions()

                results.append((a, b, a_step, best_fitness))

                if best_fitness > best_overall[0]:
                    best_overall = (best_fitness, best_sol)

    with open("woa_tuning_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "a_step", "fitness"])
        writer.writerows(results)

    print("\n=== Summary of Tuning Results ===")
    for (a, b, a_step, fitness) in results:
        print(f"a={a}, b={b}, a_step={a_step} → Fitness: {fitness:.2f}")

    print("\nBest Overall Fitness:", best_overall[0])
    print("Best Solution:", best_overall[1])


def main():
    args = parse_cl_args()

    nsols = args.nsols
    ngens = args.ngens

    # --- SUA Problem Definition with Constraints ---
    # Define your total urban land area in square meters
    total_urban_land = 1000

    # --- IMPORTANT: Set max_budget and max_water to introduce constraints ---
    # These values should be chosen to force trade-offs.
    # If you have NO budget/water constraint, set the values to float('inf')

    # Example: If you want an average of 25 liters per square meter:
    DESIRED_WATER_LIMIT_PER_SQM = 1
    MAX_WATER = DESIRED_WATER_LIMIT_PER_SQM * total_urban_land # Calculated total water limit

    MAX_BUDGET = 1 # Set to float('inf') if you don't have a budget constraint

    problem = UrbanAgProblem(
        total_land_sqm=total_urban_land,
        max_budget=MAX_BUDGET, # This will now be float('inf') if no budget
        max_water=MAX_WATER # This is your calculated total water limit
    )

    # The optimization function for WOA will be a wrapper around problem.evaluate
    # This wrapper adapts the input/output for the WhaleOptimization class.
    def sua_opt_func(solution_array):
        # If solution_array is 1D (single solution):
        if len(solution_array.shape) == 1:
            fitness, _, _, _ = problem.evaluate(solution_array)
            return fitness
        # If solution_array is 2D (multiple solutions, which WOA might pass internally):
        else:
            fitness_values = []
            for sol_row in solution_array:
                fitness, _, _, _ = problem.evaluate(sol_row)
                fitness_values.append(fitness)
            return np.array(fitness_values)

    # Constraints for WOA (taken directly from problem.bounds)
    constraints = problem.bounds

    b = args.b
    a = args.a
    a_step = a/ngens
    a_step = 0.001

    maximize = args.max # We set to True for maximizing yield

    opt_alg = WhaleOptimization(sua_opt_func, constraints, nsols, b, a, a_step, maximize)

    print(f"Starting WOA for Sustainable Urban Agriculture (Total Land: {total_urban_land} sqm)")
    print(f"Budget Constraint: ${MAX_BUDGET:.2f} (inf means no limit)")
    print(f"Water Constraint: {MAX_WATER:.2f} liters (Avg: {DESIRED_WATER_LIMIT_PER_SQM:.2f} L/sqm)")
    print(f"Optimizing to MAXIMIZE YIELD (with penalties for constraint violations)")
    print("-" * 70)

    # --- Run Optimization ---
    def log_generation(gen, fitness, solution):
        _, yield_kg, water_l, cost_usd = problem.evaluate(solution)
        print(
            f"Gen {gen + 1}: Fitness = {fitness:.2f} | Yield = {yield_kg:.2f} kg | Water = {water_l:.2f} L | Cost = ${cost_usd:.2f}")

    if args.tune_mode:
        tune_parameters(sua_opt_func, constraints)
    else:
        opt_alg.optimize(max_iters=ngens, callback=log_generation)

    # --- Final Results ---
    # Get the overall best solution from the algorithm's history
    # This handles cases where no valid solution was found too (e.g., if _best_solutions is empty)
    if not opt_alg._best_solutions:
        print("\nOptimization finished but no valid solutions were found (e.g., all solutions had infinite penalty).")
        print("Consider loosening constraints or increasing generations/solutions.")
        return # Exit if no solutions found

    final_best_fitness, final_best_solution_raw = sorted(opt_alg._best_solutions, key=lambda x:x[0], reverse=opt_alg._maximize)[0]

    # Re-evaluate the final best solution to get all metrics accurately
    _, final_yield, final_water, final_cost = problem.evaluate(final_best_solution_raw)

    # Extract and format the final solution details
    x1_perc_raw, x2_perc_raw, x3_perc_raw = final_best_solution_raw[0:3]
    c1_idx_raw, c2_idx_raw, c3_idx_raw = final_best_solution_raw[3:6]

    # Normalize percentages again for presentation (as they might not sum exactly to 100 due to WOA's continuous nature)
    total_percent_sum = x1_perc_raw + x2_perc_raw + x3_perc_raw
    if total_percent_sum > 0:
        factor = 100 / total_percent_sum
        x1_perc_adj = np.clip(x1_perc_raw * factor, 0, 100)
        x2_perc_adj = np.clip(x2_perc_raw * factor, 0, 100)
        x3_perc_adj = np.clip(x3_perc_raw * factor, 0, 100)
    else: # If all are zero
        x1_perc_adj, x2_perc_adj, x3_perc_adj = 0, 0, 0

    area_hydro = (x1_perc_adj / 100) * total_urban_land
    area_container = (x2_perc_adj / 100) * total_urban_land
    area_inground = (x3_perc_adj / 100) * total_urban_land

    # Convert crop indices back to names
    c1_name = problem.hydro_data.get(int(round(c1_idx_raw)), {'name': "Unknown Hydro Crop"})['name']
    c2_name = problem.container_data.get(int(round(c2_idx_raw)), {'name': "Unknown Container Crop"})['name']
    c3_name = problem.inground_data.get(int(round(c3_idx_raw)), {'name': "Unknown In-Ground Crop"})['name']


    print("\n" + "="*70)
    print("--- Optimal Urban Agriculture Plan (Max Yield Focus with Constraints) ---")
    print(f"Total Land Available: {total_urban_land} sqm")
    print(f"Optimal Fitness (Yield - Penalties): {final_best_fitness:.4f}")
    print(f"------------------------------------")
    print(f"Hydroponics:")
    print(f"  - Percentage of Land: {x1_perc_adj:.2f}% ({area_hydro:.2f} sqm)")
    print(f"  - Primary Crop: {c1_name}")
    print(f"Container/Raised Beds:")
    print(f"  - Percentage of Land: {x2_perc_adj:.2f}% ({area_container:.2f} sqm)")
    print(f"  - Primary Crop: {c2_name}")
    print(f"In-Ground Soil-Based Farming:")
    print(f"  - Percentage of Land: {x3_perc_adj:.2f}% ({area_inground:.2f} sqm)")
    print(f"  - Primary Crop: {c3_name}")

    print("\n--- Predicted Performance Metrics (for the Optimal Plan) ---")
    print(f"Total Annual Yield: {final_yield:.2f} kg")
    print(f"Total Annual Water Consumption: {final_water:.2f} liters")
    print(f"Total Annual Cost: ${final_cost:.2f}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()