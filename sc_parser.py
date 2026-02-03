# Parser for .sc files
import re

val_regex = r"(x|\d+)" # Regex for return values (after = or only content on a line)
str_regex = r"([se]?)\"(.*?)\"" # Regex for matching strings, groups are prefix, content
should_or = "|" # Regex that indicates that the line uses "or"
should_return = "=" # Regex that indicates that the line is a return line, expects val at the end

# A node in the full tree, represents its own rule and contains its children
class SCRulesNode:
    def __init__(self, rule: str = ""): # An empty rule is skipped
        self.rule = rule # TODO: Parse rules
        self.children = []

    def match_str(self, string: str) -> int | bool | None: # int or bool for return, None for no return
        # Check if is val
        val_match = re.fullmatch(val_regex, self.rule)
        if val_match is not None:
            value = val_match.group(1)
            if value == "x":
                return False
            else:
                return int(value)

        str_matches = re.findall(str_regex, self.rule)
        # if len(str_matches) != 0:
        is_or = should_or in self.rule
        is_return = should_return in self.rule
        is_match = False if is_or else True
        for (rule, check) in str_matches:
            if rule == "s":
                result = string.startswith(check)
            elif rule == "e":
                result = string.endswith(check)
            else:
                result = check in string
            is_match = (is_match or result) if is_or else (is_match and result)
        if is_match:
            if is_return:
                return_value = self.rule.strip().split(" ")[-1]
                if return_value == "x":
                    return False
                else:
                    return int(return_value)
            else:
                # We need to go deeper
                for child in self.children:
                    result = child.match_str(string)
                    if result is not None:
                        return result

    def __str__(self):
        return f"\"{self.rule.replace("\"", "\\\"")}\" [{",".join([str(child) for child in self.children])}]"


def build_sc_tree(sc_file: str):
    root = SCRulesNode()

    lines = [process_indent(line) for line in sc_file.split("\n")]
    parent_chain = [root]
    for line in lines:
        indent, content = line
        node = SCRulesNode(content)
        if indent > len(parent_chain)-1:
            if indent > len(parent_chain):
                raise ValueError("sc contains more indent than supported")
            parent_chain[-1].children.append(node) # Indent
            # parent_chain.append(node)
        else:
            parent_chain = parent_chain[:indent+1] # De-indent
            parent_chain[indent].children.append(node)
        parent_chain.append(node)

    return root

def process_indent(line: str) -> tuple[int, str]:
    stripped = line.lstrip()
    return len(line) - len(stripped), stripped.strip()

if __name__ == "__main__":
    tree = build_sc_tree("""s\"v\"
 \"test1\" = 0
\"test2\" = 1""")
    print(tree)
    print(tree.match_str("test1"))