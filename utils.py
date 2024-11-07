import re
import difflib

def remove_non_alphabetic(input_string):
    # need to start by removing the <br>
    cleaned_string = re.sub(r'<br>', '', input_string)
    # Use a regular expression to remove all non-alphabetic characters
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', cleaned_string)
    return cleaned_string

# return the percentage of the max portion of the psg that is included in doc
def included_overlap(psg, doc, full_inclusion=True):
    clean_psg = remove_non_alphabetic(psg)
    clean_doc = remove_non_alphabetic(doc)
    if full_inclusion:
        return 1.0 * (clean_psg in clean_doc)

    # takes quadratic time to check all possible overlaps
    s = difflib.SequenceMatcher(None,clean_doc, clean_psg, autojunk=False)
    pos_a, pos_b, size = s.find_longest_match(0, len(clean_doc), 0, len(clean_psg))
    return float(size) / len(clean_psg)

def create_unique_pid(id, count):
    return f'{id}-{count}'

def create_unique_url(id, counter):
    return f'{id}#{counter}'
