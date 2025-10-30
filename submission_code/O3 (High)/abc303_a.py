import sys

def is_similar(a: str, b: str) -> bool:
    """
    Check whether two characters are similar according to the rules:
      1. identical characters
      2. '1' and 'l' can be interchanged
      3. '0' and 'o' can be interchanged
    """
    if a == b:
        return True
    if {a, b} == {'1', 'l'}:
        return True
    if {a, b} == {'0', 'o'}:
        return True
    return False


def main() -> None:
    data = sys.stdin.read().splitlines()
    if not data:
        return
    N = int(data[0].strip())
    S = data[1].strip()
    T = data[2].strip()

    # Early exit if lengths mismatch unexpected (should not happen per constraints)
    if len(S) != N or len(T) != N:
        print("No")
        return

    for ch_s, ch_t in zip(S, T):
        if not is_similar(ch_s, ch_t):
            print("No")
            return

    print("Yes")


if __name__ == "__main__":
    main()