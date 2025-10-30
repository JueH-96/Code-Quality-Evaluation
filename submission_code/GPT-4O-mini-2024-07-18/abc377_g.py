def min_cost_to_match_strings(N, strings):
    results = []
    previous_strings = set()

    for k in range(N):
        current_string = strings[k]
        min_cost = float('inf')

        # Check cost to make current_string empty
        cost_to_empty = len(current_string)
        min_cost = min(min_cost, cost_to_empty)

        # Check cost to match any of the previous strings
        for prev_string in previous_strings:
            # Calculate the cost to convert current_string to prev_string
            len_current = len(current_string)
            len_prev = len(prev_string)
            common_length = min(len_current, len_prev)

            # Calculate the number of deletions needed
            deletions_needed = len_current - common_length
            # Calculate the number of additions needed
            additions_needed = len_prev - common_length

            total_cost = deletions_needed + additions_needed
            min_cost = min(min_cost, total_cost)

        results.append(min_cost)
        previous_strings.add(current_string)

    return results

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N = int(data[0])
    strings = data[1:N+1]
    
    results = min_cost_to_match_strings(N, strings)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()