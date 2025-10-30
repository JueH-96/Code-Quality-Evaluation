from collections import defaultdict
import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    Q = int(data[0])
    
    bag = defaultdict(int)
    results = []
    
    for i in range(1, Q + 1):
        query = data[i].split()
        if query[0] == '1':
            x = int(query[1])
            bag[x] += 1
        elif query[0] == '2':
            x = int(query[1])
            bag[x] -= 1
            if bag[x] == 0:
                del bag[x]
        elif query[0] == '3':
            results.append(len(bag))
    
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()