import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    q = int(data[0])
    global_add = 0
    groups = deque()
    output_lines = []
    index = 1
    for _ in range(q):
        parts = data[index].split()
        index += 1
        if parts[0] == '1':
            t0 = global_add
            if groups and groups[-1][0] == t0:
                last_group = groups.pop()
                groups.append((t0, last_group[1] + 1))
            else:
                groups.append((t0, 1))
        elif parts[0] == '2':
            T = int(parts[1])
            global_add += T
        elif parts[0] == '3':
            H = int(parts[1])
            X = global_add - H
            count_harvest = 0
            while groups:
                t, cnt = groups[0]
                if t <= X:
                    count_harvest += cnt
                    groups.popleft()
                else:
                    break
            output_lines.append(str(count_harvest))
    
    sys.stdout.write("
".join(output_lines))

if __name__ == "__main__":
    main()