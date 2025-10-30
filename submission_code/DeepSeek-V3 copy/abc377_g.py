# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1:N+1]
    
    # Initialize a dictionary to store the minimum cost for each string
    # We will use a trie to efficiently find the minimum cost
    # However, for simplicity, we will use a dictionary to store the strings and their costs
    # Since the constraints are large, we need an efficient way to find the minimum cost
    
    # We will use a dictionary to store the strings and their costs
    # For each string, we will find the minimum cost to make it match any previous string or empty
    
    # Initialize the list to store the results
    results = []
    
    # Initialize a set to store the previous strings
    previous_strings = set()
    
    for i in range(N):
        current_string = S[i]
        min_cost = len(current_string)  # Cost to make it empty
        
        # Iterate through all previous strings to find the minimum cost
        for prev_string in previous_strings:
            # Find the longest common prefix
            lcp = 0
            while lcp < len(current_string) and lcp < len(prev_string) and current_string[lcp] == prev_string[lcp]:
                lcp += 1
            # Calculate the cost
            cost = (len(current_string) - lcp) + (len(prev_string) - lcp)
            if cost < min_cost:
                min_cost = cost
        
        # Add the current string to the set of previous strings
        previous_strings.add(current_string)
        
        # Append the result
        results.append(min_cost)
    
    # Print the results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()