import sys

def min_cost_to_make_string(T, N, bags):
    """
    Calculate the minimum cost to make the final string S equal to T.

    Args:
    T (str): The target string.
    N (int): The number of bags.
    bags (list): A list of lists, where each sublist contains strings in a bag.

    Returns:
    int: The minimum cost to make the final string S equal to T. If it's impossible, return -1.
    """
    # Initialize a dictionary to store the minimum cost for each prefix of T
    min_cost = {0: 0}

    # Iterate over each bag
    for i in range(N):
        # Initialize a temporary dictionary to store the updated minimum cost
        temp = {}
        # Iterate over each string in the current bag
        for string in bags[i]:
            # Iterate over each prefix of T
            for j in range(len(T) - len(string), -1, -1):
                # If the string matches the suffix of the prefix
                if T[j:j + len(string)] == string:
                    # Update the minimum cost for the new prefix
                    if j + len(string) not in temp or temp[j + len(string)] > (min_cost.get(j, float('inf')) + 1):
                        temp[j + len(string)] = min_cost.get(j, float('inf')) + 1
        # Update the minimum cost dictionary
        min_cost.update(temp)

    # Return the minimum cost to make the final string S equal to T
    return min_cost.get(len(T), -1)

def main():
    # Read the input from stdin
    T = input().strip()
    N = int(input().strip())
    bags = []
    for _ in range(N):
        bag = input().strip().split()[1:]
        bags.append(bag)

    # Calculate and print the minimum cost
    print(min_cost_to_make_string(T, N, bags))

if __name__ == "__main__":
    main()