import sys

def main():
    # Read input from stdin
    R = int(sys.stdin.read().strip())

    # Determine the current number of ^ based on the rating
    if 1 <= R <= 99:
        current_level = 1
    elif 100 <= R <= 199:
        current_level = 2
    elif 200 <= R <= 299:
        current_level = 3

    # Calculate the minimum increase required to reach the next level
    if current_level == 1:
        next_level_threshold = 100
    elif current_level == 2:
        next_level_threshold = 200
    elif current_level == 3:
        next_level_threshold = 300

    increase_needed = next_level_threshold - R

    # Print the result to stdout
    print(increase_needed)

if __name__ == "__main__":
    main()