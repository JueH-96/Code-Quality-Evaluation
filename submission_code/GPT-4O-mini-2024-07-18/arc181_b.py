def can_form_equal_strings(t, cases):
    results = []
    for S, X, Y in cases:
        count_X0 = X.count('0')
        count_X1 = X.count('1')
        count_Y0 = Y.count('0')
        count_Y1 = Y.count('1')
        
        # Check if the number of S's added in both cases can match
        if count_X0 == count_Y0 and count_X1 == count_Y1:
            results.append("Yes")
        else:
            results.append("No")
    
    return results

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    t = int(data[0])
    cases = []
    
    for i in range(t):
        S = data[3 * i + 1]
        X = data[3 * i + 2]
        Y = data[3 * i + 3]
        cases.append((S, X, Y))
    
    results = can_form_equal_strings(t, cases)
    print("
".join(results))

if __name__ == "__main__":
    main()