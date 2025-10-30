import sys
import threading
def main():
    import sys
    data = sys.stdin.readline().split()
    if not data:
        data = sys.stdin.readline().split()
    n = int(data[0]); x = int(data[1]); y = int(data[2])
    s = sys.stdin.readline().rstrip()
    t = sys.stdin.readline().rstrip()
    # If already equal
    if s == t:
        print("Yes")
        return
    # If no operation ever possible
    if x + y > n:
        print("No")
        return
    # build prefix sums of ones for s and t
    # O_s[i] = number of '1' in s[0:i]
    O_s = [0] * (n+1)
    O_t = [0] * (n+1)
    for i,ch in enumerate(s):
        O_s[i+1] = O_s[i] + (ch == '1')
    for i,ch in enumerate(t):
        O_t[i+1] = O_t[i] + (ch == '1')
    # check if s has any A or B operation possible
    limit = n - (x + y)
    s_has_op = False
    for i in range(limit+1):
        # Operation A: 0^x 1^y
        if O_s[i+x] - O_s[i] == 0 and O_s[i+x+y] - O_s[i+x] == y:
            s_has_op = True
            break
        # Operation B: 1^y 0^x
        if O_s[i+y] - O_s[i] == y and O_s[i+x+y] - O_s[i+y] == 0:
            s_has_op = True
            break
    # check t has any op possible
    t_has_op = False
    for i in range(limit+1):
        if O_t[i+x] - O_t[i] == 0 and O_t[i+x+y] - O_t[i+x] == y:
            t_has_op = True
            break
        if O_t[i+y] - O_t[i] == y and O_t[i+x+y] - O_t[i+y] == 0:
            t_has_op = True
            break
    # if either is isolated (no op) then cannot transform unless equal,
    # but we already checked s==t
    if not s_has_op or not t_has_op:
        print("No")
        return
    # Check zero‐counts per class mod y
    zcnt_s = [0] * y
    zcnt_t = [0] * y
    for i,ch in enumerate(s):
        if ch == '0':
            zcnt_s[i % y] += 1
    for i,ch in enumerate(t):
        if ch == '0':
            zcnt_t[i % y] += 1
    if zcnt_s != zcnt_t:
        print("No")
        return
    # Check one‐counts per class mod x
    ocnt_s = [0] * x
    ocnt_t = [0] * x
    for i,ch in enumerate(s):
        if ch == '1':
            ocnt_s[i % x] += 1
    for i,ch in enumerate(t):
        if ch == '1':
            ocnt_t[i % x] += 1
    if ocnt_s != ocnt_t:
        print("No")
        return
    # passed all checks
    print("Yes")

if __name__ == "__main__":
    main()