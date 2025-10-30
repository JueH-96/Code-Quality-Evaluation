import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1

    events = []
    for _ in range(N):
        t = int(input[ptr])
        x = int(input[ptr+1])
        events.append((t, x))
        ptr +=2

    # Compute min_k for each x
    min_k = [0] * (N + 1)
    for x in range(1, N+1):
        current_block = 0
        max_block = 0
        for event in events:
            t, typ = event
            if typ == x:
                if t == 1:
                    current_block +=1
                else:
                    if current_block > max_block:
                        max_block = current_block
                    current_block = 0
        # Check the last block
        if current_block > max_block:
            max_block = current_block
        min_k[x] = max_block

    # Check if possible
    possible = True
    for x in range(1, N+1):
        a_x = events.count((1, x))
        b_x = events.count((2, x))
        if a_x < b_x:
            possible = False
            break
        if b_x > 0 and min_k[x] < b_x:
            possible = False
            break

    if not possible:
        print(-1)
        return

    K_min = max(min_k[x] for x in range(1, N+1))

    # Now, process the selected array
    selected = [False] * (N+1)
    for x in range(1, N+1):
        current_block = []
        for event in events:
            t, typ = event
            if typ == x and t == 1:
                current_block.append(event)
            elif typ == x and t == 2:
                for idx in current_block:
                    selected[idx] = True
                current_block = []
        # After processing all events, check if current_block is not empty
        if current_block:
            for idx in current_block:
                selected[idx] = True

    # Now, construct the output
    print(K_min)
    output = ['0'] * (N)
    for i in range(N):
        t = events[i][0]
        if t == 1:
            if selected[i]:
                output[i] = '1'
    print(' '.join(output))

if __name__ == '__main__':
    main()