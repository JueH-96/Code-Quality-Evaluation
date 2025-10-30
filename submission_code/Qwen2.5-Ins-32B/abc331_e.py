# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    
    a = list(map(int, data[3:N+3]))
    b = list(map(int, data[N+3:N+M+3]))
    
    c = list(map(int, data[N+M+3::2]))
    d = list(map(int, data[N+M+4::2]))
    
    max_a = max(a)
    max_b = max(b)
    
    max_price = max_a + max_b
    
    for i in range(L):
        if a[c[i]-1] + b[d[i]-1] == max_price:
            max_price = -1
            break
    
    if max_price == -1:
        a.remove(max_a)
        b.remove(max_b)
        max_a = max(a)
        max_b = max(b)
        max_price = max_a + max_b
    
    print(max_price)

if __name__ == "__main__":
    main()