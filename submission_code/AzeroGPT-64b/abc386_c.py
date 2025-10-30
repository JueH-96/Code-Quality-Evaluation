k=int(input())
s=input()
t=input()

idx_s=0
idx_t=0
# find initial same letters numbers
while idx_t < len(t) and idx_s < len(s) and t[idx_t] == s[idx_s]:
    idx_t += 1
    idx_s += 1

# trying to erase or insert
is_possible = 0
if len(s) == idx_s and len(t) == idx_t:
    # trying match all with 0 operation
    is_possible = 1

elif len(t) - len(s) + idx_t  == idx_s:
    # initial string match only and erase
    is_possible = 1
elif len(t) - len(s) == idx_t  - idx_s:
    # initial string match only and insert
    is_possible = 1
elif min(len(s), len(t)) -max(idx_t, idx_s) == 1:
    # last char do not match so replace
    is_possible = 1
else:
    # check middle search
    idx_s_m = idx_s
    idx_t_m = idx_t
    while idx_t_m < len(t) and idx_s_m < len(s) and t[idx_t_m] == s[idx_s_m]:
        idx_t_m += 1
        idx_s_m += 1

    if len(t) - len(s) + idx_t_m == idx_s_m:
        is_possible = 1
    elif len(t) - len(s)  == idx_t_m- idx_s_m:
        is_possible= 1

print("Yes" if is_possible else "No")