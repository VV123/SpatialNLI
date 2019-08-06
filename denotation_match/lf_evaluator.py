import tempfile
import subprocess
import os
import re
from data import *

# YOU SHOULD NOT NEED TO LOOK AT THIS FILE.
# This file consists of evaluation code adapted from Jia + Liang, wrapping predictions and sending them to a Java
# backend for evaluation against the knowledge base.

# Find the top-scoring derivation that executed without error
def pick_derivations(all_pred_dens, all_derivs, is_error_fn):
    derivs = []
    pred_dens = []
    cur_start = 0
    if len(all_pred_dens) == 0:
        print(len(all_derivs))
        print("No legal derivations! Likely you're getting an error when calling the evaluation in Java")
        for deriv_set in all_derivs:
            derivs.append(Derivation("", 0.0, [""]))
            pred_dens.append("Example FAILED TO PARSE")
        return (derivs, pred_dens)

    for deriv_set in all_derivs:
        # What to do if 0?
        for i in range(len(deriv_set)):
            cur_denotation = all_pred_dens[cur_start + i]
            if not is_error_fn(cur_denotation):
                derivs.append(deriv_set[i])
                pred_dens.append(cur_denotation)
                break
        else:
            if len(deriv_set) == 0:
                # Try to avoid crashing
                derivs.append(Derivation("", 0.0, [""]))
                pred_dens.append("Example FAILED TO PARSE")
            else:
                derivs.append(deriv_set[0])  # Default to first derivation
                pred_dens.append(all_pred_dens[cur_start])
        cur_start += len(deriv_set)
    return (derivs, pred_dens)


class GeoqueryDomain(object):
    def postprocess_lf(self, lf):
        # Undo the variable name standardization.
        cur_var = chr(ord('A') - 1)
        toks = lf.split(' ')
        new_toks = []
        for w in toks:
            if w == 'NV':
                cur_var = chr(ord(cur_var) + 1)
                new_toks.append(cur_var)
            elif w.startswith('V'):
                ind = int(w[1:])
                new_toks.append(chr(ord(cur_var) - ind))
            else:
                new_toks.append(w)
        return ' '.join(new_toks)

    def clean_name(self, name):
        return name.split(',')[0].replace("'", '').strip()

    def format_lf(self, lf):
        # Strip underscores, collapse spaces when not inside quotation marks
        lf = self.postprocess_lf(lf)
        toks = []
        in_quotes = False
        quoted_toks = []
        for t in lf.split():
            if in_quotes:
                if t == "'":
                    in_quotes = False
                    toks.append('"%s"' % ' '.join(quoted_toks))
                    quoted_toks = []
                else:
                    quoted_toks.append(t)
            else:
                if t == "'":
                    in_quotes = True
                else:
                    if len(t) > 1 and t.startswith('_'):
                        toks.append(t[1:])
                    else:
                        toks.append(t)
        lf = ''.join(toks)
        # Balance parentheses
        num_left_paren = sum(1 for c in lf if c == '(')
        num_right_paren = sum(1 for c in lf if c == ')')
        diff = num_left_paren - num_right_paren
        if diff > 0:
            lf = lf + ')' * diff
        return lf

    def get_denotation(self, line):
        m = re.search('\{[^}]*\}', line)
        if m:
            return m.group(0)
        else:
            return line.strip()

    def print_failures(self, dens, name):
        num_syntax_error = sum(d == 'Example FAILED TO PARSE' for d in dens)
        num_exec_error = sum(d == 'Example FAILED TO EXECUTE' for d in dens)
        num_join_error = sum('Join failed syntactically' in d for d in dens)
        print('%s: %d syntax errors, %d executor errors' % (
            name, num_syntax_error, num_exec_error))

    def is_error(self, d):
        return 'FAILED' in d or 'Join failed syntactically' in d

    def compare_answers(self, true_answers, rcds, all_derivs):
        all_lfs = ([self.format_lf(s) for s in true_answers] +
                [self.format_lf(' '.join(d.y_toks))
                for x in all_derivs for d in x])
        tf_lines = ['_parse([query], %s).' % lf for lf in all_lfs]
        tf = tempfile.NamedTemporaryFile(suffix='.dlog')
        for line in tf_lines:
            tf.write(line.encode() + b'\n')
            #print(line)
        tf.flush()
        # JAVA INVOCATION: uncomment the following three lines to print the java code output and stop there if you
        # need to check if the Java is working
        #####
        # msg = subprocess.check_output(['evaluator/geoquery', tf.name], stderr=subprocess.STDOUT)
        # print(repr(msg.decode("utf-8"))
        #exit()
        #####
        try:
            msg = subprocess.check_output(['evaluator/geoquery', tf.name]).decode("utf-8")
            # Use this line instead if the subprocess call is crashing
            # msg = ""
        except subprocess.CalledProcessError as err:
            print("Error in subprocess Geoquery evaluation call. Command output:")
            print(err.output)
            print(err.returncode)
            exit()
        tf.close()
        denotations = [self.get_denotation(line)
                       for line in msg.split('\n')
                       if line.startswith('        Example')]
        true_dens = denotations[:len(true_answers)]
        with open('denotations.txt', 'w') as f:
            f.write('\n'.join(denotations))
        if len(true_dens) == 0:
            true_dens = ["" for i in range(0, len(true_answers))]
        all_pred_dens = denotations[len(true_answers):]

        # Find the top-scoring derivation that executed without error
        derivs, pred_dens = pick_derivations(all_pred_dens, all_derivs, self.is_error)
        self.print_failures(true_dens, 'gold')
        self.print_failures(pred_dens, 'predicted')
        num = 0
        for t, p, (a, b) in zip(true_dens, pred_dens, rcds):
            #print('%s: %s == %s' % (t == p, t, p))
            if t == p and a != b:
                num += 1
                print(a)
                print(b)
                print(t)
                print(p)
                print('------------------------')
        print('Percent of denotation match but not exact match --> {0:d}/{1:d} = {2:.2f}'.format(num, len(rcds), num*1./len(rcds)))
        return derivs, [t == p for t, p in zip(true_dens, pred_dens)]


##########################
# UNUSED IN THIS PROJECT #
##########################
# Evaluation code for the Overnight domains adapted from Robin Jia and Percy Liang.
class OvernightEvaluator(object):
    def format_lf(self, lf):
        replacements = [
            ('! ', '!'),
            ('SW', 'edu.stanford.nlp.sempre.overnight.SimpleWorld'),
        ]
        for a, b in replacements:
            lf = lf.replace(a, b)
        # Balance parentheses
        num_left_paren = sum(1 for c in lf if c == '(')
        num_right_paren = sum(1 for c in lf if c == ')')
        diff = num_left_paren - num_right_paren
        if diff > 0:
            while len(lf) > 0 and lf[-1] == '(' and diff > 0:
                lf = lf[:-1]
                diff -= 1
            if len(lf) == 0: return ''
            lf = lf + ' )' * diff
        return lf

    def is_error(self, d):
        return 'BADJAVA' in d or 'ERROR' in d or d == 'null'

    def compare_answers(self, true_answers, all_derivs):
        # Put all "true" answers at the start of the list, then add all derivations that
        # were produced by decoding
        all_lfs = ([self.format_lf(s) for s in true_answers] +
                   [self.format_lf(' '.join(d.y_toks))
                    for x in all_derivs for d in x])
        tf_lines = all_lfs
        tf = tempfile.NamedTemporaryFile(suffix='.examples')
        for line in tf_lines:
            tf.write(line.encode() + b'\n')
            #print(line)
        tf.flush()
        f = open(tf.name)
        subdomain = "calendar" # TODO: set subdomain
        msg = subprocess.check_output(['evaluator/overnight', subdomain, tf.name])
        tf.close()
        print(len(all_lfs))
        denotations = [line.split('\t')[1] for line in msg.decode("utf-8").split('\n')
                       if line.startswith('targetValue\t')]
        print(len(denotations))
        print(len(true_answers))
        true_dens = denotations[:len(true_answers)]
        all_pred_dens = denotations[len(true_answers):]
        derivs, pred_dens = pick_derivations(all_pred_dens, all_derivs, self.is_error)
        return derivs, [t == p for t, p in zip(true_dens, pred_dens)]
