import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    runs = []
    current_run = []
    i = 0
    while i < N - 1:
        if A[i] == A[i + 1]:
            current_run.append(A[i])
            i += 2
        else:
            if current_run:
                runs.append(current_run)
                current_run = []
            i += 1
    if current_run:
        runs.append(current_run)

    max_len = 0
    for run in runs:
        seen = {}
        left = 0
        current_max = 0
        for right in range(len(run)):
            val = run[right]
            if val in seen:
                if seen[val] >= left:
                    left = seen[val] + 1
            seen[val] = right
            current_len = right - left + 1
            if current_len > current_max:
                current_max = current_len
        if current_max * 2 > max_len:
            max_len = current_max * 2
    print(max_len)

if __name__ == "__main__":
    main()