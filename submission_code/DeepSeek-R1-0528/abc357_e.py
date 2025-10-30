import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a_list = list(map(int, data[1:1+n]))
    
    state = [0] * (n + 1)
    cnt = [0] * (n + 1)
    
    for i in range(1, n + 1):
        if state[i] == 0:
            path = []
            cur = i
            while state[cur] == 0:
                state[cur] = 1
                path.append(cur)
                cur = a_list[cur - 1]
            
            if state[cur] == 1:
                idx = path.index(cur)
                cycle_size = len(path) - idx
                for j in range(idx, len(path)):
                    cnt[path[j]] = cycle_size
                for j in range(idx - 1, -1, -1):
                    cnt[path[j]] = 1 + cnt[a_list[path[j] - 1]]
            else:
                for j in range(len(path) - 1, -1, -1):
                    cnt[path[j]] = 1 + cnt[a_list[path[j] - 1]]
            
            for node in path:
                state[node] = 2
                
    total = sum(cnt[1:])
    print(total)

if __name__ == "__main__":
    main()