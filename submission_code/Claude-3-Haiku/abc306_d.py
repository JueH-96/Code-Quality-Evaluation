from typing import List

def max_tastiness(N: int, X_Y: List[List[int]]) -> int:
    """
    Finds the maximum possible sum of tastiness of the courses that Takahashi eats.
    
    Args:
        N (int): The number of courses.
        X_Y (List[List[int]]): A list of lists, where each inner list contains the values of X_i and Y_i for the i-th course.
    
    Returns:
        int: The maximum possible sum of tastiness of the courses that Takahashi eats.
    """
    # Initialize the state to healthy
    state = 0
    
    # Initialize the total tastiness to 0
    total_tastiness = 0
    
    for x, y in X_Y:
        # If the current course is antidotal
        if x == 0:
            # If the current state is healthy, eat the course
            if state == 0:
                total_tastiness += y
            # If the current state is upset, eat the course to become healthy
            elif state == 1:
                total_tastiness += y
                state = 0
        # If the current course is poisonous
        elif x == 1:
            # If the current state is healthy, become upset
            if state == 0:
                state = 1
            # If the current state is upset, die
            elif state == 1:
                return total_tastiness
    
    # If Takahashi makes it out of the restaurant alive, return the total tastiness
    return total_tastiness