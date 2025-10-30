import sys

def in_pos(s, t, x):
    if s == t:
        return False
    if s < t:
        return s < x < t
    else:
        return x > s or x < t

def in_neg(s, t, x):
    if s == t:
        return False
    if s > t:
        return t < x < s
    else:
        return x > t or x < s

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    a = list(map(int, data[2:2+n]))
    b = list(map(int, data[2+n:2+2*n]))
    
    total_cost = 0
    for i in range(n):
        d = (b[i] - a[i]) % m
        cost0 = d
        cost1 = m - d
        
        valid0 = True
        valid1 = True
        
        if i >= 1:
            if in_pos(a[i], b[i], b[i-1]):
                valid0 = False
            if in_neg(a[i], b[i], b[i-1]):
                valid1 = False
                
        if i < n-1:
            if in_pos(a[i], b[i], a[i+1]):
                valid0 = False
            if in_neg(a[i], b[i], a[i+1]):
                valid1 = False
                
        if not valid0 and not valid1:
            print(-1)
            return
            
        if valid0 and valid1:
            cost = min(cost0, cost1)
        elif valid0:
            cost = cost0
        else:
            cost = cost1
            
        total_cost += cost
        
    print(total_cost)

if __name__ == "__main__":
    main()