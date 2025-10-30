def solve():
    t = int(input())
    for _ in range(t):
        s = input()
        x = input()
        y = input()
        
        # If lengths are different, impossible to match
        if len(x) * len(s) != len(y) * len(s):
            print("No")
            continue
            
        # Count number of 0s and 1s in each string
        x0 = x.count('0')
        x1 = x.count('1')
        y0 = y.count('0')
        y1 = y.count('1')
        
        # If number of S's (0s) are different, impossible
        if x0 != y0:
            print("No")
            continue
            
        # If number of T's (1s) are different, impossible
        if x1 != y1:
            print("No")
            continue
            
        # Get positions of S and T in both strings
        pos_x = []
        pos_y = []
        
        for i in range(len(x)):
            if x[i] == '0':
                pos_x.extend(list(range(i*len(s), (i+1)*len(s))))
                
        for i in range(len(y)):
            if y[i] == '0':
                pos_y.extend(list(range(i*len(s), (i+1)*len(s))))
                
        # Check if S appears at same positions in both strings
        if pos_x == pos_y:
            print("Yes")
        else:
            print("No")

solve()