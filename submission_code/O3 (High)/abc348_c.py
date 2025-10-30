import sys
from collections import defaultdict

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # store the minimum deliciousness for each color
    min_by_color = {}
    
    for _ in range(n):
        a = int(next(it))
        c = int(next(it))
        if c in min_by_color:
            if a < min_by_color[c]:
                min_by_color[c] = a
        else:
            min_by_color[c] = a
    
    # the answer is the largest among those minima
    answer = max(min_by_color.values())
    print(answer)

if __name__ == "__main__":
    main()