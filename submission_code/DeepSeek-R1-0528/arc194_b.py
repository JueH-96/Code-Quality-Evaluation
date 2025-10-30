import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    
    fenw = [0] * (n + 1)
    
    def update(index, delta):
        while index <= n:
            fenw[index] += delta
            index += index & -index
            
    def query(index):
        s = 0
        while index > 0:
            s += fenw[index]
            index -= index & -index
        return s

    total_cost = 0
    for i in range(n - 1, -1, -1):
        num = P[i]
        if num <= n:
            count = query(num - 1)
            total_cost += (i + 1) * count
            update(num, 1)
            
    print(total_cost)

if __name__ == "__main__":
    main()