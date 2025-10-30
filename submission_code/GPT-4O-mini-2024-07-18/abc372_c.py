def count_abc_substrings(s):
    count = 0
    for i in range(len(s) - 2):
        if s[i:i + 3] == "ABC":
            count += 1
    return count

def process_queries(n, q, s, queries):
    results = []
    current_string = list(s)  # Convert to list for mutability
    abc_count = count_abc_substrings(current_string)

    for x, c in queries:
        # Adjust for 0-based index
        index = x - 1
        if current_string[index] != c:  # Only update if there's a change
            # Update the character
            current_string[index] = c
            
            # Recalculate the number of "ABC" substrings
            abc_count = count_abc_substrings(current_string)

        results.append(abc_count)

    return results

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    n, q = map(int, data[0].split())
    s = data[1]
    queries = [tuple(data[i + 2].split()) for i in range(q)]
    queries = [(int(x), c) for x, c in queries]

    results = process_queries(n, q, s, queries)
    print("
".join(map(str, results)))

if __name__ == "__main__":
    main()