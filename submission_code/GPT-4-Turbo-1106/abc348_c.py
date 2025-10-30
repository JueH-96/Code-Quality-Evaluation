import sys
from collections import defaultdict

def main():
    input_lines = sys.stdin.readlines()
    n = int(input_lines[0])
    beans = defaultdict(list)
    
    for line in input_lines[1:]:
        deliciousness, color = map(int, line.split())
        beans[color].append(deliciousness)
    
    max_min_deliciousness = 0
    for color in beans:
        max_min_deliciousness = max(max_min_deliciousness, min(beans[color]))
    
    print(max_min_deliciousness)

if __name__ == "__main__":
    main()