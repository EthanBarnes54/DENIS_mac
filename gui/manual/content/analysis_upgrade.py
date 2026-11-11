"""Authored manual pages for the Analysis upgrade (2026-11).

Three pages: the upgrade toggles and the original method each replaces;
the statistical methods and systematics; nuclear moments and charge
radii. Every statement is checked against the code it names.

Depends on: gui.manual.render.
"""

from gui.manual import render as R


def an_upgrades():
    return f"""
<h1>Analysis Upgrades: Toggles and Their Originals</h1>

<p>Each upgrade to the Analysis tab is a switch that is <b>off by default</b>.
Off, or absent from an older session, gives the original method byte for
byte: a parity check reproduces the arrays every fit sees with all of them
off. The Source-block switches are listed in
{R.code_ref("gui/analysis/pipeline.py: UPGRADE_TOGGLES")}.</p>

<h2>Source block (Advanced Data Settings)</h2>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Control</th><th>On</th><th>Off = the original method</th></tr>
<tr><td>Cooler correction <code>rolling</code>, with its number of
points (default 100)</td><td>each bunch's cooler voltage is the mean of
the readings around it, which removes the meter's scatter</td>
<td><code>pbp</code>: each event's own reading; <code>mean</code>: the
run's average</td></tr>
<tr><td>Laser: <code>wavemeter mean</code></td><td>the wavemeter averaged
over the run, from ORACLE's monitor record beside it</td>
<td><code>header</code>: the run's recorded setpoint (also for a run
without a monitor record)</td></tr>
<tr><td>Use the run's recorded dividers</td><td>the scan and cooler
divider ratios ORACLE recorded for the run</td><td>clstools' fixed
dividers</td></tr>
<tr><td>Dead time [µs], per PMT; <b>Suggested dead times</b> fills in
ORACLE's suggestion, never by itself</td><td>a count that follows a count
on the same PMT, in the same bunch, within that time is dropped before
binning (afterpulses, and the real photons that land as close)</td>
<td>every count kept</td></tr>
<tr><td>Per-run cooler / laser override (right-click a file)</td>
<td>that run's value replaces its header's</td><td>the header</td></tr>
</table>

{R.warning("The dead-time gate suits PMTs that see well under one photon "
           "a bunch. At higher rates it also removes real photons and leaves "
           "the counts below Poisson. The gate records, per PMT, the share "
           "of real photons lost at the strongest step "
           "(<code>real_loss_strongest</code> in the run's "
           "<code>dead_time</code> metadata). Read it before trusting a "
           "gated run.")}

<h2>Output block (Diagnostics)</h2>
<ul>
<li><b>Run quality</b>: each run's variance over mean, for all PMTs and
each, with the excess split into what the PMTs share (the ions per bunch)
and what is a PMT's own (afterpulsing); ORACLE's verdict and failed checks
from its performance report; the cooler scatter when <code>rolling</code>
is used. "!" when variance/mean is above 1.2 or a timing check failed. In
the fit report and the Isotope Shifts run preview.</li>
<li><b>Validate fit</b>, with its number of refits: spectra drawn from the
fit's own model, refitted the same way, and per parameter the pull width
and the 68 % / 95 % coverage. The fit's own results are untouched.
Single-run fits only; not with emcee.</li>
</ul>
{R.tip("A pull width near 1 means the quoted errors are right. Widths "
       "near 0.7 with a likelihood fit mean its error bars are the "
       "uncorrected ones, about √2 too large: tick the Fitter block's "
       "correction.")}

<h2>Systematics and Moments &amp; Radii</h2>
<p>The Systematics tab's <b>Sources</b> table and <b>Propagate sources</b>
add a general error budget beside the cooler-offset scan; the Moments &amp;
Radii tab turns fitted constants and shifts into μ, Q and δ⟨r²⟩. With no
source added and the tab unused, nothing changes. See
{R.link("physics-statistics-systematics")} and
{R.link("physics-moments-radii")}.</p>

<h2>Reading ORACLE runs</h2>
<p>Runs taken with ORACLE carry a record of how they were taken. The Source
block's <b>Meta</b> ends with an ORACLE section (version, commit, rack, scan
type, region, reference flag, the calibration's verdict and checks, the
bench-table fit), and the ASDF Viewer documents the record. When <b>Add
files</b> on a Sample project brings in runs ORACLE tagged as reference
runs, DENIS asks once whether to move them to a Reference project; it never
moves them by itself.</p>

{R.related(["an-source-block", "an-output-block", "physics-fit-objectives",
            "tool-asdf-viewer"])}
"""


def physics_statistics_systematics():
    return f"""
<h1>Statistical Methods and Systematics</h1>

<h2>Fit statistics</h2>
<p>Chi-square with √(y+1) errors is biased at low counts; the Poisson
likelihood is not (Baker &amp; Cousins, Nucl. Instrum. Methods 221 (1984)
437). Its numerical error bars come out √2 too large unless the Fitter
block's correction is ticked; see {R.link("physics-fit-objectives")}.</p>

<h2>Overdispersion</h2>
<p>For Poisson counts the variance of a step's counts over the scans equals
their mean. <b>Run quality</b> reports variance/mean <i>D</i> from each
step's repeats, with the significance of <i>D</i> − 1 from the Poisson
null, and splits a PMT's excess into the part all PMTs share (more or fewer
ions per bunch) and its own (afterpulsing). The dead-time gate removes the
latter. {R.code_ref("cls_estimations/statistics.py: Overdispersion")}</p>

<h2>Weighted means and the Birge ratio</h2>
<p>The Isotope Shifts tab averages runs with weights 1/σ² and scales the
error by the Birge ratio √(χ²/(N−1)) when it exceeds 1 (Birge, Phys. Rev.
40 (1932) 207).</p>

<h2>Validate fit</h2>
<p>A parametric bootstrap: spectra drawn from the best fit (at the run's
measured <i>D</i> when Run quality ran), refitted with the same settings.
For each parameter, the pull (refit − truth)/σ should have mean 0 and width
1, and the ±1σ / ±2σ intervals should cover the truth 68 % / 95 % of the
time. {R.code_ref("cls_estimations/statistics.py: Pull_Coverage")}</p>

<h2>The systematic budget</h2>
<p>Each source in the Systematics tab's <b>Sources</b> table is a 1σ width
on one perturbation of the apparatus (cooler offset or divider, scan
divider, scan-axis offset or gain, laser offset), with a unit, a kind
(measured, declared, unknown) and a correlation class: <code>run</code>,
<code>campaign</code>, <code>region</code> or <code>table</code>.</p>
<ul>
<li><b>±1σ responses.</b> <b>Propagate sources</b> refits the whole chain
(reference fit, GP, corrections, sample fits, shifts) with each source at
+1σ and −1σ; a source's effect is the change it makes.</li>
<li><b>Over runs.</b> A source that moves every run at once enters the
weighted mean as the weighted mean of the run responses, sign kept; a
per-run source as their weighted quadrature sum
({R.code_ref("cls_estimations/systematics.py: Combine_Runs")}).</li>
<li><b>Monte Carlo</b> (on request, a number of draws): the shared sources
drawn together from N(0, σ) for every run at once; per-run sources still
enter from their ±1σ responses.</li>
<li><b>In shifts</b>, a source common to sample and reference cancels
where both runs respond alike; the budget shows each shift's response, so
the cancellation is visible rather than assumed.</li>
</ul>
<p>The method follows A. Raggio's review of the analysis (November 2026);
the GP drift correction follows van den Borne (2025).</p>

{R.related(["an-isotope-shifts", "an-reference-gp", "physics-gp",
            "an-upgrades"])}
"""


def physics_moments_radii():
    return f"""
<h1>Nuclear Moments and Charge Radii</h1>

<p>The <b>Moments &amp; Radii</b> tab, beside Isotope Shifts and Systematics
in the Analysis tab, turns fitted hyperfine constants and isotope shifts
into nuclear moments and changes in mean-square charge radius.
{R.code_ref("cls_estimations/nuclear_moments.py")}</p>

<h2>Moments by the ratio method</h2>
<p>Against a reference isotope of known spin, moment and constants on the
same line:</p>
<p style="text-align:center">μ = μ<sub>ref</sub> · (I / I<sub>ref</sub>) ·
(A / A<sub>ref</sub>), &nbsp;&nbsp; Q = Q<sub>ref</sub> · (B /
B<sub>ref</sub>)</p>
<p>each from the lower and the upper level where measured, and their mean
with the fit's correlation of A<sub>l</sub> and A<sub>u</sub>. The
hyperfine anomaly is neglected unless a percentage is given, which enters
as an error. The A<sub>u</sub>/A<sub>l</sub> ratio check compares the
measured ratio with the reference's. Errors are kept apart as statistical,
systematic and reference.</p>

<h2>Charge radii and the King plot</h2>
<p>The isotope shift between A and A′ is</p>
<p style="text-align:center">δν<sup>AA′</sup> = F · δ⟨r²⟩<sup>AA′</sup> +
K / μ<sub>AA′</sub>, &nbsp;&nbsp; 1/μ<sub>AA′</sub> = (m<sub>A′</sub> −
m<sub>A</sub>) / (m<sub>A</sub> m<sub>A′</sub>)</p>
<p>with the masses of the ions in the charge state measured (atomic masses
less the electrons). So δ⟨r²⟩ = (δν − K/μ)/F. <b>F and K</b> come from one
of three modes:</p>
<ul>
<li><b>F, K given</b>, with σ and their correlation;</li>
<li><b>King-plot calibration</b>: a York fit (York et al., Am. J. Phys. 72
(2004) 367) of the modified shifts μδν against μδ⟨r²⟩ for pairs of known
δ⟨r²⟩, errors in both;</li>
<li><b>Second line</b>: shifts of the same pairs on a line of known F and K
(W. H. King, <i>Isotope Shifts in Atomic Spectra</i>, Plenum, 1984).</li>
</ul>
<p>Shifts against one reference isotope share its error, so the δ⟨r²⟩ are
correlated; the table gives statistical, systematic and atomic (F, K)
errors and the tab keeps the full covariance. An isomer pair has
1/μ = 0: its shift is F · δ⟨r²⟩ alone. It is marked as an isomer in the
δ⟨r²⟩ table and left off the King plot, whose axes are multiplied by μ.
<b>Send to Results</b> writes the tables and the King plot as an
iteration.</p>

{R.related(["physics-isotope-shift", "physics-ab-scaling", "an-isotope-shifts",
            "physics-statistics-systematics"])}
"""


PAGES = {
    "an-upgrades": an_upgrades,
    "physics-statistics-systematics": physics_statistics_systematics,
    "physics-moments-radii": physics_moments_radii,
}
