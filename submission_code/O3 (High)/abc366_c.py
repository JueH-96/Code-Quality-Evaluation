import sys
from collections import defaultdict

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    Q = int(next(it))
    
    counter = defaultdict(int)      # x -> current multiplicity
    distinct = 0                    # how many different x are in the bag
    out_lines = []
    
    for _ in range(Q):
        t = next(it)                # query type (as bytes)
        if t == b'1':               # insert
            x = int(next(it))
            if counter[x] == 0:
                distinct += 1
            counter[x] += 1
        elif t == b'2':             # delete
            x = int(next(it))
            counter[x] -= 1
            if counter[x] == 0:
                distinct -= 1
                # optional: del counter[x]        # keep dict small
        else:                       # t == b'3'
            out_lines.append(str(distinct))
    
    sys.stdout.write('
'.join(out_lines))

if __name__ == "__main__":
    main()