# Annotation guidelines
JATS is a risotto... the annotation is not systematic and stable so here the need for this document. 

## Acknowledgement 
Usually acknoledgement is contained within `<ack>`, however could happen that:
 - contains information about the copyright or licencing
 - contains a mix betweeen funding, copyright, conflict of interests, acknoledgment, etc...
 - is hidden under the carpet `<fn` (TBC)
 - is stored under `<sec` 

### <p @content-type

When it's under `<ack>` we tag the <p> with the `@content-type`

Examples: 

```xml
  <back>
    <ack>
      <p content-type="acknowledgement">The authors gratefully acknowledge Kirkstall Ltd., [...].</p>
      <p>The authors have declared no conflict of interest.</p>
    </ack>
```

```xml
    <ack>
      <title>Acknowledgment</title>
      <p content-type="acknowledgement">The authors wish to thank the Department of Science and Technology (DST), West Bengal, India, for the financial assistance.</p>
    </ack>
```

```xml
<ack>
  <title>Acknowledgment</title>
  <p content-type="acknowledgement">This research was supported by University of California, Davis through Academic Federation Research Grant Program&#x2014;Innovative Developmental Award.</p>
  <p><bold>Open Access</bold> This article is distributed under the terms of the Creative Commons Attribution Noncommercial License which permits any noncommercial use, distribution, and reproduction in any medium, provided the original author(s) and source are credited.</p>
</ack>
```

```xml
<ack>
      <p content-type="funding">This work was supported by the Caffeine Working Group of the North American Branch of the International Life Sciences Institute (ILSI). ILSI North America is a public, nonprofit foundation that provides a forum to advance the understanding of scientific issues related to the nutritional quality and safety of the food supply by sponsoring research programs, educational seminars, workshops, and publications. ILSI North America receives support primarily from its industry membership and secondarily receives grants from governmental agencies. Three other scientists were contacted by the Caffeine Committee of ILSI to review the recent epidemiological studies dealing with the reproductive and developmental effects of caffeine (Peck et al., <xref ref-type="bibr" rid="b174">2010</xref>). These epidemiologists reviewed the results of caffeine exposure in studies concerned with pregnancy loss (SAs and miscarriage), congenital malformations (CMs), growth retardation, stillbirth, prematurity, and fertility.</p>
      <p content-type="acknowlegement">Drs. Brent and Christian received unrestricted grants from ILSI for their work reviewing, analyzing, and summarizing the information contained in this article. Dr. Brent was an academic board member of HESI for 18 years, a division of ILSI. Academic appointees receive no compensation for their services. The authors have not participated as expert witnesses in caffeine litigation and have no conflict of interest with regard to the material presented in this publication. The opinions expressed herein are those of the authors and do not necessarily represent the views of ILSI North America.</p>
      <p content-type="acknowledgement">The authors gratefully acknowledge the interaction with the TERIS program and their consultants, specifically with regard to the evaluation of the epidemiology studies of the reproductive and developmental effects of caffeine. Twenty-five years ago TERIS was funded by a grant from the HHS department and it is now based at the University of Washington as a section of the Department of Pediatrics as a non-profit program. The members of the TERIS evaluation consultant group are:</p>
```

### <sec @sec-type

```xml
<sec sec-type="acknowledgement">
  <title>Acknowledgements</title>
  <p>We thank Prof. B&#xE4;rbel Hahn-H&#xE4;gerdal for her valuable comments, Catherine Paul for critically reading the manuscript and Birgit Johansson for providing fresh soil for DNA extraction. This work was financed by the Swedish Research Council (VR).</p>
</sec>
```


## Funding

Funding is usually under the acknoledgement, but in general it's annotated at level of `<p content-type="funding"`.

```xml
<ack>
    <p content-type="funding">This work was supported by the Caffeine Working Group of the North American Branch of the International Life Sciences Institute (ILSI). ILSI North America is a public, nonprofit foundation that provides a forum to advance the understanding of scientific issues related to the nutritional quality and safety of the food supply by sponsoring research programs, educational seminars, workshops, and publications. ILSI North America receives support primarily from its industry membership and secondarily receives grants from governmental agencies. Three other scientists were contacted by the Caffeine Committee of ILSI to review the recent epidemiological studies dealing with the reproductive and developmental effects of caffeine (Peck et al., <xref ref-type="bibr" rid="b174">2010</xref>). These epidemiologists reviewed the results of caffeine exposure in studies concerned with pregnancy loss (SAs and miscarriage), congenital malformations (CMs), growth retardation, stillbirth, prematurity, and fertility.</p>
    <p content-type="acknowlegement">Drs. Brent and Christian received unrestricted grants from ILSI for their work reviewing, analyzing, and summarizing the information contained in this article. Dr. Brent was an academic board member of HESI for 18 years, a division of ILSI. Academic appointees receive no compensation for their services. The authors have not participated as expert witnesses in caffeine litigation and have no conflict of interest with regard to the material presented in this publication. The opinions expressed herein are those of the authors and do not necessarily represent the views of ILSI North America.</p>
    <p content-type="acknowledgement">The authors gratefully acknowledge the interaction with the TERIS program and their consultants, specifically with regard to the evaluation of the epidemiology studies of the reproductive and developmental effects of caffeine. Twenty-five years ago TERIS was funded by a grant from the HHS department and it is now based at the University of Washington as a section of the Department of Pediatrics as a non-profit program. The members of the TERIS evaluation consultant group are:</p>
```

```xml
<fn-group>
  <fn fn-type="financial-disclosure">
    <p content-type="funding">This work was supported by the <funding-source id="gs1">National Institute of Neurological Disease and Stroke</funding-source> [grant number <award-id rid="gs1">NS43419</award-id>-01A]. Salary support for G.H.D.V. was provided by a Research Career Service Award from the <funding-source id="gs2">Department of Veteran Affairs</funding-source>.</p>
  </fn>
</fn-group>
```

## Availability statements

### <p @content-type 

This is the default method that target the `<p>` tag containing the text. 

```xml
<sec>
<p content-type="availability">....</p>
</sec>
```

```xml
<fn-group>
  <fn id="fnu1">
    <p content-type="availability">Supplementary data and figures for this paper are available from the IUCr electronic archives (Reference: <ext-link ext-link-type="uri" xlink:href="http://scripts.iucr.org/cgi-bin/sendsup?im2283">IM2283</ext-link>).</p>
  </fn>
</fn-group>
```

### <sec @specific-use

There are cases where the content of the availability statement is not included completeley within `<p>` tags

Example: 

```xml
<sec sec-type="supplementary-material" specific-use="availability">
  <title>Supporting information</title>
  <p>The following supplementary materials are available for this article:</p>
  <p><bold>Table S1.</bold> Included studies.</p>
  <supplementary-material content-type="local-data" id="SD1">
    <media xlink:href="bjo0118-0671-SD1.doc" xlink:type="simple" id="d32e5236" position="anchor" mimetype="application" mime-subtype="msword"/>
  </supplementary-material>
  <p><bold>Table S2.</bold> Reporting of weight measurements.</p>
  <supplementary-material content-type="local-data" id="SD2">
    <media xlink:href="bjo0118-0671-SD2.doc" xlink:type="simple" id="d32e5242" position="anchor" mimetype="application" mime-subtype="msword"/>
  </supplementary-material>
  <p>Additional Supporting Information may be found in the online version of this article.</p>
  <p>Please note: Wiley-Blackwell are not responsible for the content or functionality of any supporting information supplied by the authors. Any queries (other than missing material) should be directed to the corresponding author.</p>
</sec>
```
