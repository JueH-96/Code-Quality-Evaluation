import sys

def min_slimes(N, slimes):
    slimes.sort()
    result = 0
    for size, count in reversed(slimes):
        result += count % 2
        if count > 1:
            next_count = count // 2
            next_size = size * 2
            inserted = False
            for i in range(N):
                if slimes[i][0] == next_size:
                    slimes[i] = (next_size, slimes[i][1] + next_count)
                    inserted = True
                    break
            if not inserted:
                slimes.append((next_size, next_count))
                slimes.sort()
    return result

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    slimes = []
    for i in range(1, 2*N+1, 2):
        size = int(data[i])
        count = int(data[i+1])
        slimes.append((size, count))
    
    print(min_slimes(N, slimes))

if __name__ == "__main__":
    main()