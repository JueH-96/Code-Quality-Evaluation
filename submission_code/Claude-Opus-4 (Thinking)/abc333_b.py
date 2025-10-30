# YOUR CODE HERE
def vertex_to_num(v):
    return ord(v) - ord('A')

def distance_type(v1, v2):
    i = vertex_to_num(v1)
    j = vertex_to_num(v2)
    diff = abs(i - j)
    return min(diff, 5 - diff)

s1s2 = input().strip()
t1t2 = input().strip()

s1, s2 = s1s2[0], s1s2[1]
t1, t2 = t1t2[0], t1t2[1]

dist_s = distance_type(s1, s2)
dist_t = distance_type(t1, t2)

if dist_s == dist_t:
    print("Yes")
else:
    print("No")