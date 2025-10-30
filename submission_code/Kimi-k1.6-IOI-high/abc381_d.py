def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    pairs = []
    for i in range(len(A) - 1):
        if A[i] == A[i+1]:
            pairs.append((A[i], i))
    
    runs = []
    current_run = []
    prev_start = None
    for pair in pairs:
        val, start = pair
        if not current_run:
            current_run.append(pair)
            prev_start = start
        else:
            if start == prev_start + 2:
                current_run.append(pair)
                prev_start = start
            else:
                runs.append(current_run)
                current_run = [pair]
                prev_start = start
    if current_run:
        runs.append(current_run)
    
    max_unique = 0
    for run in runs:
        values = [p[0] for p in run]
        seen = {}
        left = 0
        current_max = 0
        for right in range(len(values)):
            val = values[right]
            if val in seen:
                if seen[val] >= left:
                    left = seen[val] + 1
            seen[val] = right
            current_max = max(current_max, right - left + 1)
        max_unique = max(max_unique, current_max)
    
    print(max_unique * 2)

if __name__ == '__main__':
    main()