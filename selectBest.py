import random


def selectBestFromAspirants(aspirants):
    """
    Select the best individual from the aspirants:
    1. First, sort based on rank.
    2. If multiple individuals have the same rank, sort based on crowding distance.
    3. If there are still ties in rank and crowding distance, select the individual with the smallest tree depth.
    4. If tree depth is also identical, randomly select one individual.
    """

    # Sort by rank ascending, and crowding distance descending
    aspirants.sort(key=lambda ind: (ind.fitness.rank, -ind.fitness.crowding_dist))

    # Get the individuals with the best (lowest) rank
    best_rank = aspirants[0].fitness.rank
    candidates = [ind for ind in aspirants if ind.fitness.rank == best_rank]

    # If only one individual has the best rank, return it directly
    if len(candidates) == 1:
        return candidates[0]

    # Further filter based on crowding distance
    candidates.sort(key=lambda ind: -ind.fitness.crowding_dist)
    best_crowding_dist = candidates[0].fitness.crowding_dist
    candidates = [ind for ind in candidates if ind.fitness.crowding_dist == best_crowding_dist]

    # If only one individual has the best rank and crowding distance, return it directly
    if len(candidates) == 1:
        return candidates[0]

    # Further filter based on tree depth
    candidates.sort(key=lambda ind: len(ind))
    smallest_gp_tree = candidates[0]

    # If only one individual has the smallest tree depth, return it directly
    if len([ind for ind in candidates if len(ind) == len(smallest_gp_tree)]) == 1:
        return smallest_gp_tree

    # If tree depth is also identical, randomly select one individual
    return random.choice(candidates)