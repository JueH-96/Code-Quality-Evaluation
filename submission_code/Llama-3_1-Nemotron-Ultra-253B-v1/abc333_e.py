import sys

def main():
    N = int(sys.stdin.readline())
    events = []
    for _ in range(N):
        t, x = map(int, sys.stdin.readline().split())
        events.append((t, x))
    
    taken = [False] * N
    required = [0] * (N + 2)  # x can be up to N
    
    for i in reversed(range(N)):
        t, x = events[i]
        if t == 2:
            required[x] += 1
        else:
            if required[x] > 0:
                taken[i] = True
                required[x] -= 1
            else:
                taken[i] = False
    
    # Check if any required is still positive
    for x in range(1, N + 1):
        if required[x] > 0:
            print(-1)
            return
    
    # Simulate to find max_total and actions
    total = 0
    max_total = 0
    actions = []
    for i in range(N):
        t, x = events[i]
        if t == 1:
            if taken[i]:
                total += 1
                if total > max_total:
                    max_total = total
                actions.append(1)
            else:
                actions.append(0)
        else:
            total -= 1
    
    print(max_total)
    print(' '.join(map(str, actions)))

if __name__ == '__main__':
    main()