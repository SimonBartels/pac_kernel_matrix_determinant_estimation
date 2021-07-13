import numpy as np
from tikzplotlib import clean_figure, get_tikz_code
from matplotlib import pyplot as plt
from tikzplotlib_fix.tikzplotlib_fix import fix_tikzplotlib_clean_figure


fix_tikzplotlib_clean_figure()


def get_displayed_length_scale(parameter_ls: float):
    """
    Takes the length-scale parameter and returns the value that we display.

    The parameter is on log-scale. The kernels store the length-scale squared. That's why we need an additional function
    here.

    :param parameter_ls: parameter value for the length-scale
    :return:
        the value to be displayed
    """
    return np.exp(parameter_ls)


def write_tikz_file(file_name, gca=None, do_clean_figure=True, legend_name=None, style=None):
    if gca is None:
        gca = plt.gca()
    gca.set_title('')
    if do_clean_figure:
        clean_figure()

    extra_axis_parameters = ['xticklabel style = {align=center}']
    if legend_name is not None:
        extra_axis_parameters.append('legend to name=%s' % legend_name)
    if style is not None:
        extra_axis_parameters.append(style)
    tikz_code = get_tikz_code(axis_width='\\figwidth', axis_height='\\figheight',
                              extra_axis_parameters=extra_axis_parameters)
    if legend_name is not None:
        tikz_code = enable_legend_entry_refs(tikz_code)
    tikz_code = tikz_code.replace('\\$', '$')
    tikz_code = tikz_code.replace('ylabel={\\labely{}},\n',
                                  'ylabel={\\labely{}},\n'
                                  'ymajorticks={\\ticklabelsy},\n'
                                  'xmajorticks={\\ticklabelsx},\n')
    tikz_code = tikz_code.replace('\\end{axis}', '\\clearlegend{};\n\\end{axis}')
    f = open(file_name, 'w+')
    f.write(tikz_code)
    f.close()


def enable_legend_entry_refs(tikz_code: str):
    addlegendentry = "addlegendentry{"
    substrs = tikz_code.split(addlegendentry)
    tikz_code = substrs[0]
    for i in range(1, len(substrs)):
        label_name = substrs[i][:substrs[i].find('}') + 2]
        # we assume here that label names of the form \command{}
        assert(label_name[0] == '\\' and label_name[-2:] == '}}')
        tikz_code += u"label{leg:" + label_name[1:-3] + "}"
        tikz_code += "\\" + addlegendentry + substrs[i]
    return tikz_code


def _rs_to_str(rs):
    """
    Since mlflow stores the precisions as strings it can be tricky to find them there...
    """
    if rs[0] == 0.0:
        assert(np.all(np.array(rs) == 0.0))
        return ["%.1f" % r for r in rs]
    return ["%.6f" % r for r in rs]
