class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        while 'a' in s:
            s = s.replace('a', '', 1)
        while 'b' in s:
            s = s.replace('b', '', 1)
        while 'c' in s:
            s = s.replace('c', '', 1)
        while 'd' in s:
            s = s.replace('d', '', 1)
        while 'e' in s:
            s = s.replace('e', '', 1)
        while 'f' in s:
            s = s.replace('f', '', 1)
        while 'g' in s:
            s = s.replace('g', '', 1)
        while 'h' in s:
            s = s.replace('h', '', 1)
        while 'i' in s:
            s = s.replace('i', '', 1)
        while 'j' in s:
            s = s.replace('j', '', 1)
        while 'k' in s:
            s = s.replace('k', '', 1)
        while 'l' in s:
            s = s.replace('l', '', 1)
        while 'm' in s:
            s = s.replace('m', '', 1)
        while 'n' in s:
            s = s.replace('n', '', 1)
        while 'o' in s:
            s = s.replace('o', '', 1)
        while 'p' in s:
            s = s.replace('p', '', 1)
        while 'q' in s:
            s = s.replace('q', '', 1)
        while 'r' in s:
            s = s.replace('r', '', 1)
        while 's' in s:
            s = s.replace('s', '', 1)
        while 't' in s:
            s = s.replace('t', '', 1)
        while 'u' in s:
            s = s.replace('u', '', 1)
        while 'v' in s:
            s = s.replace('v', '', 1)
        while 'w' in s:
            s = s.replace('w', '', 1)
        while 'x' in s:
            s = s.replace('x', '', 1)
        while 'y' in s:
            s = s.replace('y', '', 1)
        while 'z' in s:
            s = s.replace('z', '', 1)
        return s