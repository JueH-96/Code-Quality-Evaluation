from itertools import product

def solve(v1, v2, v3):
    for a1, b1, c1, a2, b2, c2, a3, b3, c3 in product(range(-100, 101), repeat=9):
        c1 = (a1, b1, c1)
        c2 = (a2, b2, c2)
        c3 = (a3, b3, c3)
        
        # Check if the volumes match the given constraints
        if (
            volume(c1) + volume(c2) + volume(c3) - volume(intersect(c1, c2)) - volume(intersect(c1, c3)) - volume(intersect(c2, c3)) + volume(intersect(c1, c2, c3)) == v1 and
            volume(intersect(c1, c2)) + volume(intersect(c1, c3)) + volume(intersect(c2, c3)) - volume(intersect(c1, c2, c3)) == v2 and
            volume(intersect(c1, c2, c3)) == v3
        ):
            return f"Yes
{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}"
    return "No"

def volume(cube):
    a, b, c = cube
    return 7 * 7 * 7

def intersect(c1, c2, c3=None):
    a1, b1, c1 = c1
    a2, b2, c2 = c2
    if c3 is None:
        return (max(a1, a2), max(b1, b2), max(c1, c2)), (min(a1 + 7, a2 + 7), min(b1 + 7, b2 + 7), min(c1 + 7, c2 + 7))
    a3, b3, c3 = c3
    return (max(a1, a2, a3), max(b1, b2, b3), max(c1, c2, c3)), (min(a1 + 7, a2 + 7, a3 + 7), min(b1 + 7, b2 + 7, b3 + 7), min(c1 + 7, c2 + 7, c3 + 7))

if __name__ == "__main__":
    v1, v2, v3 = map(int, input().split())
    print(solve(v1, v2, v3))