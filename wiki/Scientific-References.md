# Scientific References & Bibliography

The numerical algorithms, continuous peak models, background decomposition techniques, and baseline filters implemented in `python-cmat` are based on the following literature in semiconductor $\gamma$-ray spectroscopy and signal processing:

---

1. **Du, P., Kibbe, W. A., & Lin, S. M.** (2006).  
   *"Improved peak detection in mass spectrometry by incorporating continuous wavelet transform-based pattern matching"*.  
   *Bioinformatics*, 22(17), 2059–2065.  
   [DOI: 10.1093/bioinformatics/btl355](https://doi.org/10.1093/bioinformatics/btl355)  
   *(Foundational algorithm for multi-scale Continuous Wavelet Transform ridge detection and noise estimation in counting spectra).*

2. **Gamba, E. R., Bruce, A. M., & Rudigier, M.** (2019).  
   *"Treatment of background in $\gamma$-$\gamma$ fast-timing measurements"*.  
   *Nuclear Instruments and Methods in Physics Research Section A*, 928, 93–103.  
   [DOI: 10.1016/j.nima.2019.03.028](https://doi.org/10.1016/j.nima.2019.03.028)  
   *(Mathematical formulation of the 4-component coincidence background decomposition, ridge extraction, and Peak-to-Total-Background ratio $\Pi$).*

3. **Morhác, M., Kliman, J., Jandel, M., Krupa, L., & Matoušek, V.** (1997).  
   *"Study of background and peak decomposition in multidimensional coincidence $\gamma$-ray spectra"*.  
   *Nuclear Instruments and Methods in Physics Research Section A*, 401(1), 113–131.  
   [DOI: 10.1016/S0168-9002(97)01023-1](https://doi.org/10.1016/S0168-9002(97)01023-1)  
   *(Foundational background estimation using LLS transformation and decreasing-order SNIP in ROOT TSpectrum and gamma-ray coincidence analysis).*

4. **Phillips, G. W., & Marlow, K. W.** (1976).  
   *"Automatic analysis of gamma-ray spectra from germanium detectors"*.  
   *Nuclear Instruments and Methods*, 137(3), 525–536.  
   [DOI: 10.1016/0029-554X(76)90472-X](https://doi.org/10.1016/0029-554X(76)90472-X)  
   *(Original formulation of the Hypermet peak shape: Gaussian + convolved exponential tail + erfc step function).*

5. **Campbell, J. L., & Maxwell, J. A.** (1993).  
   *"Analytical representation of Si(Li) and HPGe detector response functions"*.  
   *Nuclear Instruments and Methods in Physics Research Section B*, 73(4), 545–551.  
   [DOI: 10.1016/0168-583X(93)95837-K](https://doi.org/10.1016/0168-583X(93)95837-K)  
   *(Comprehensive evaluation of analytical detector response functions and physical origin of peak tailing components).*

6. **Radford, D. C.** (1995).  
   *"ESCL8R and LEVIT8R: Software for interactive analysis of HPGe coincidence data sets"*.  
   *Nuclear Instruments and Methods in Physics Research Section A*, 361(1-2), 297–305.  
   [ScienceDirect Link](https://www.sciencedirect.com/science/article/abs/pii/0168900295001832)  
   *(Foundational reference for the RadWare analysis package and the `gf3` peak fitting algorithms used throughout modern nuclear structure coincidence analysis).*

7. **Helmer, R. G., & Lee, M. A.** (1980).  
   *"Analytical functions for fitting peaks from Ge semiconductor detectors"*.  
   *Nuclear Instruments and Methods*, 178(2-3), 499–512.  
   [DOI: 10.1016/0029-554X(80)90830-7](https://doi.org/10.1016/0029-554X(80)90830-7)  
   *(Systematic comparison of analytical peak fitting formulations for germanium semiconductor detectors).*

8. **Routti, J. T., & Prussin, S. G.** (1969).  
   *"Photopeak method for the computer analysis of gamma-ray spectra from semiconductor detectors"*.  
   *Nuclear Instruments and Methods*, 72(2), 125–142.  
   [DOI: 10.1016/0029-554X(69)90148-7](https://doi.org/10.1016/0029-554X(69)90148-7)  
   *(Original formulation of the Gaussian peak with continuous exponential tails in the SAMPO program).*

9. **Ryan, C. G., Clayton, E., Griffin, W. L., Sie, S. H., & Cousens, D. R.** (1988).  
   *"SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications"*.  
   *Nuclear Instruments and Methods in Physics Research Section B*, 34(3), 396–402.  
   [DOI: 10.1016/0168-583X(88)90063-8](https://doi.org/10.1016/0168-583X(88)90063-8)  
   *(Original formulation of the Statistics-sensitive Non-linear Iterative Peak-clipping [SNIP] background estimation algorithm).*

10. **Mariscotti, M. A.** (1967).  
    *"A method for automatic identification of peaks in the presence of background"*.  
    *Nuclear Instruments and Methods*, 50(2), 309–320.  
    [DOI: 10.1016/0029-554X(67)90058-4](https://doi.org/10.1016/0029-554X(67)90058-4)  
    *(Pioneering method for automated peak detection and continuum background estimation using generalized second-difference filtering in GASPware `trackn.F`).*

11. **Morhác, M.** (2009).  
    *"Sophisticated algorithms of analysis of spectrometric data"*.  
    *Nuclear Instruments and Methods in Physics Research Section A*, 600(2), 478–487.  
    [DOI: 10.1016/j.nima.2008.11.132](https://doi.org/10.1016/j.nima.2008.11.132)  
    *(Comprehensive analysis of peak clipping, multi-scale smoothing, deconvolution, and multidimensional background estimation in gamma-ray spectroscopy).*

12. **Knoll, G. F.** (2010).  
    *"Radiation Detection and Measurement"*, 4th Edition, John Wiley & Sons, New York.  
    ISBN: 978-0-470-13148-0.  
    *(Chapters 12 & 18: Germanium gamma-ray detectors, pulse height defect, hole trapping, and peak shape asymmetry).*
