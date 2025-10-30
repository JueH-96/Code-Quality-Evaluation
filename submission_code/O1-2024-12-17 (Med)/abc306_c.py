def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Build a list to store the positions of each element i (1 <= i <= N)
    occurrences = [[] for _ in range(N + 1)]
    for idx, val in enumerate(A, start=1):
        occurrences[val].append(idx)
    
    # Compute f(i) for each i
    # f(i) is the second occurrence (middle) among the three positions where i appears
    fi_pairs = []
    for i in range(1, N + 1):
        mid_idx = occurrences[i][1]
        fi_pairs.append((mid_idx, i))
    
    # Sort the pairs by the middle occurrence
    fi_pairs.sort(key=lambda x: x[0])

    # Output the sequence of elements i in ascending order of f(i)
    answer = [str(pair[1]) for pair in fi_pairs]
    print(" ".join(answer))

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()