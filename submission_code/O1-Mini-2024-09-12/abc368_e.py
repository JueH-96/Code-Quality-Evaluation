# YOUR CODE HERE
import sys
import sys
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    import sys

    N, M, X1 = map(int, sys.stdin.readline().split())
    trains = []
    for j in range(1, M+1):
        A, B, S, T = map(int, sys.stdin.readline().split())
        trains.append({'idx': j, 'A': A, 'B': B, 'S': S, 'T': T})

    # Create list_in[c] for each city c
    list_in = [[] for _ in range(N+1)]
    for train in trains:
        list_in[train['B']].append(train)

    # Sort list_in[c] by T ascending
    for c in range(1, N+1):
        list_in[c].sort(key=lambda x: x['T'])

    # Sort trains by S ascending
    sorted_trains = sorted(trains, key=lambda x: x['S'])

    # Initialize X[j] for j=1 to M
    X = [0] * (M+1)  # 1-based indexing
    X[1] = X1

    # Initialize max_current[c] and pointers[c]
    max_current = [0] * (N+1)
    pointers = [0] * (N+1)

    for train in sorted_trains:
        j = train['idx']
        A_j = train['A']
        S_j = train['S']
        B_j = train['B']
        T_j = train['T']

        # Process incoming trains to A_j with T_i <= S_j
        while pointers[A_j] < len(list_in[A_j]) and list_in[A_j][pointers[A_j]]['T'] <= S_j:
            i = list_in[A_j][pointers[A_j]]['idx']
            max_current[A_j] = max(max_current[A_j], X[i] + list_in[A_j][pointers[A_j]]['T'])
            pointers[A_j] +=1

        if j !=1:
            X[j] = max(0, max_current[A_j] - S_j)
        
    # Now, to process X[j] correctly by updating max_current after setting X[j]
    # Re-sort the trains to process in sorted order again
    sorted_trains = sorted(trains, key=lambda x: x['S'])
    max_current = [0] * (N+1)
    pointers = [0] * (N+1)

    for train in sorted_trains:
        j = train['idx']
        A_j = train['A']
        S_j = train['S']
        B_j = train['B']
        T_j = train['T']

        # Process incoming trains to A_j with T_i <= S_j
        while pointers[A_j] < len(list_in[A_j]) and list_in[A_j][pointers[A_j]]['T'] <= S_j:
            i = list_in[A_j][pointers[A_j]]['idx']
            max_current[A_j] = max(max_current[A_j], X[i] + list_in[A_j][pointers[A_j]]['T'])
            pointers[A_j] +=1

        if j !=1:
            X[j] = max(0, max_current[A_j] - S_j)
        else:
            X[j] = X1

        # After setting X[j], it can affect future trains
        max_current[B_j] = max(max_current[B_j], X[j] + T_j)

    # Prepare the output
    output = []
    for j in range(2, M+1):
        output.append(str(X[j]))
    print(' '.join(output))

if __name__ == "__main__":
    main()