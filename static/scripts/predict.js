let spectrumIndex = 0
let predicting = false

let predInd = 0

let unit = 'ppm'
let unitLabel = 'ppm'

let labels = {
  ppm: 'Chemical shift [ppm]',
  Hz: 'Frequency [Hz]'
}

let expnoParams = null

let data

let preds

function ClosestIndex (arr, x) {
  let i
  let i0 = 0
  for (i = 0; i < arr.length; i++) {
    if (Math.abs(arr[i] - x) < Math.abs(arr[i0] - x)) {
      i0 = i
    }
  }
  return i0
}

// Next/previous controls
function ChangeSpectrum (n) {
  ShowSpectrum(spectrumIndex += n)
}

function ShowSpectrum (n) {
  const spectra = document.getElementsByClassName('spectrum')

  if (n >= spectra.length) {
    spectrumIndex = 0
  } else if (n < 0) {
    spectrumIndex = spectra.length - 1
  }

  for (let i = 0; i < spectra.length; i++) {
    spectra[i].style.display = 'none'
  }
  spectra[spectrumIndex].style.display = 'block'

  let ymax = 0
  for (let i = 0; i < data.length; i++) {
    ymax = Math.max(ymax, Math.max(...data[i].yr))
  }

  const disp = document.getElementById(`spectrum_fig${spectrumIndex + 1}`)

  const xmin = Number(document.getElementById('rangel').value)
  const xmax = Number(document.getElementById('ranger').value)

  function PlotSpectrum (spectrumIndex) {
    Plotly.newPlot(
      disp,
      [{
        x: data[spectrumIndex][unit],
        y: data[spectrumIndex].yr.map(
          function (item) {
            return item / ymax
          }
        )
      }],
      {
        xaxis: {
          showgrid: false,
          showline: true,
          range: [xmin, xmax],
          zeroline: false,
          title: { text: labels[unitLabel] }
        },
        yaxis: {
          showgrid: false,
          showline: true,
          range: [0, 1.5],
          zeroline: false,
          showticklabels: false,
          title: {
            text: 'Normalized intensity [a.u.]'
          }
        },
        margin: {
          l: 40,
          r: 20,
          t: 20,
          b: 40
        }
      }
    )
  }

  PlotSpectrum(spectrumIndex)
}

function UpdateRange () {
  ShowSpectrum(spectrumIndex)
}

function UpdateUnits () {
  const units = document.getElementsByName('unit')
  let i
  for (i = 0; i < units.length; i++) {
    if (units[i].checked) {
      // Get current range
      const rl = document.getElementById('rangel')
      const rr = document.getElementById('ranger')
      const xmin = Number(rl.value)
      const xmax = Number(rr.value)

      // Convert range and update unit
      const imin = ClosestIndex(data[0][unit], xmin)
      const imax = ClosestIndex(data[0][unit], xmax)

      unit = units[i].value
      unitLabel = units[i].value

      if (document.getElementById('acqu2').checked) {
        unit += '2'
      }

      rl.value = Number(Math.round(data[0][unit][imin]))
      rr.value = Number(Math.round(data[0][unit][imax]))

      document.getElementById('range-unit').innerHTML = ` ${unitLabel} `

      // Update plots
      ShowSpectrum(spectrumIndex)
    }
  }
}

function ChangePred (n) {
  PlotPred(document.getElementById('evolving-pred'), predInd += n)
}

function PlotPred (elem, n) {
  if (n >= preds.preds.length) {
    predInd = 0
  } else if (n < 0) {
    predInd = preds.preds.length - 1
  }

  const xmin = Number(document.getElementById('rangel').value)
  const xmax = Number(document.getElementById('ranger').value)

  const specIdx = predInd + preds.specs.length - preds.preds.length

  const predY = preds.preds[predInd].map(
    function (item) {
      return item / Math.max(...preds.preds[predInd])
    }
  )
  const predDy = preds.err[predInd].map(
    function (item) {
      return item / Math.max(...preds.preds[predInd])
    }
  )
  const predYmax = predY.map((e, i) => e + predDy[i])
  const predYmin = predY.map((e, i) => e - predDy[i])

  Plotly.newPlot(
    elem,
    [
      {
        x: preds[unitLabel],
        y: preds.specs[specIdx].map(
          function (item) {
            return item / Math.max(...preds.specs[specIdx])
          }
        ),
        hoverinfo: 'x+y',
        name: `${preds.wrs[specIdx]} KHz MAS spectrum`
      },
      {
        x: preds[unitLabel],
        y: predY,
        hoverinfo: 'x+y',
        name: 'Predicted PIP spectrum'
      },
      {
        x: preds[unitLabel],
        y: predYmax,
        fill: 'tonexty',
        type: 'scatter',
        mode: 'none',
        fillcolor: 'rgba(255, 127, 14, 0)',
        hoverinfo: 'x',
        showlegend: false
      },
      {
        x: preds[unitLabel],
        y: predYmin,
        fill: 'tonexty',
        type: 'scatter',
        mode: 'none',
        fillcolor: 'rgba(255, 127, 14, 0.2)',
        hoverinfo: 'x',
        name: 'Predicted PIP uncertainty'
      }
    ],
    {
      xaxis: {
        showgrid: false,
        showline: true,
        range: [xmin, xmax],
        zeroline: false,
        title: { text: labels[unitLabel] }
      },
      yaxis: {
        showgrid: false,
        showline: true,
        range: [0, 1.5],
        zeroline: false,
        showticklabels: false,
        title: { text: 'Normalized intensity [a.u.]' }
      },
      margin: {
        l: 40,
        r: 20,
        t: 20,
        b: 40
      },
      legend: {
        yanchor: 'top',
        y: 0.99,
        xanchor: 'right',
        x: 0.99
      }
    }
  )
}

function PlotPredAll (elem, dy) {
  let ymax = 1e-12
  for (let i = 0; i < preds.specs.length; i++) {
    ymax = Math.max(ymax, Math.max(...preds.specs[i]))
  }

  const xmin = Number(document.getElementById('rangel').value)
  const xmax = Number(document.getElementById('ranger').value)

  const specsToPlot = []
  const c0 = [0, 255, 255]
  const dc = [0, -255, 0]
  let y0 = dy
  for (let i = 0; i < preds.specs.length; i++) {
    const col = c0.map((e, k) => e + dc[k] * i / (preds.specs.length - 1))

    specsToPlot.push({
      x: preds[unitLabel],
      y: preds.specs[i].map(function (item) { return y0 + (item / ymax) }),
      hoverinfo: 'x',
      name: `${preds.wrs[i]} KHz MAS spectrum`,
      line: { color: `rgb(${col[0]}, ${col[1]}, ${col[2]})` },
      showlegend: (i === 0 || i === preds.specs.length - 1)
    })
    y0 += dy
  }

  const predY = preds.preds[preds.preds.length - 1].map(
    function (item) {
      return item / Math.max(...preds.preds[preds.preds.length - 1])
    }
  )
  const predDy = preds.err[preds.preds.length - 1].map(
    function (item) {
      return item / Math.max(...preds.preds[preds.preds.length - 1])
    }
  )
  const predYmax = predY.map((e, i) => e + predDy[i])
  const predYmin = predY.map((e, i) => e - predDy[i])

  specsToPlot.push(
    {
      x: preds[unitLabel],
      y: predY,
      hoverinfo: 'x+y',
      name: 'Predicted PIP spectrum',
      line: { color: 'rgb(255, 127, 14)' }
    }
  )
  specsToPlot.push(
    {
      x: preds[unitLabel],
      y: predYmax,
      fill: 'tonexty',
      type: 'scatter',
      mode: 'none',
      fillcolor: 'rgba(255, 127, 14, 0)',
      hoverinfo: 'x',
      showlegend: false
    }
  )
  specsToPlot.push(
    {
      x: preds[unitLabel],
      y: predYmin,
      fill: 'tonexty',
      type: 'scatter',
      mode: 'none',
      fillcolor: 'rgba(255, 127, 14, 0.2)',
      hoverinfo: 'x',
      name: 'Predicted PIP uncertainty'
    }
  )

  Plotly.newPlot(
    elem,
    specsToPlot,
    {
      xaxis: {
        showgrid: false,
        showline: true,
        range: [xmin, xmax],
        zeroline: false,
        title: { text: labels[unitLabel] }
      },
      yaxis: {
        showgrid: false,
        showline: true,
        range: [0, 2 + y0],
        zeroline: false,
        showticklabels: false,
        title: {
          text: 'Normalized intensity [a.u.]'
        }
      },
      margin: {
        l: 40,
        r: 20,
        t: 20,
        b: 40
      },
      legend: {
        yanchor: 'top',
        y: 0.99,
        xanchor: 'right',
        x: 0.99
      }
    }
  )
}

function UpdateSelMas () {
  const wrMin = Number(document.getElementById('mas-rangel').value)
  const wrMax = Number(document.getElementById('mas-ranger').value)

  for (let i = 0; i < data.length; i++) {
    if (Number(document.getElementById(`mas_fig${i + 1}`).value) >= wrMin && Number(document.getElementById(`mas_fig${i + 1}`).value) <= wrMax) {
      document.getElementById(`select_${i + 1}`).checked = true
    } else {
      document.getElementById(`select_${i + 1}`).checked = false
    }
  }
}

function HasValidProcno (are2d, expno, procnos) {
  let hasValid = false
  for (const procno of procnos) {
    if (are2d[`${expno}-${procno}`] !== null) {
      hasValid = true
      break
    }
  }
  return hasValid
}

function GetPossibleProcnos () {
  // Don't trigger if the list of expnos is still in construction
  if (expnoParams == null) {
    return
  }

  const possibleProcnos = []
  const allProcnos = []
  const allExpnos = []
  for (const [expno, procnos] of Object.entries(expnoParams.procnos)) {
    for (const procno of procnos) {
      if (!allProcnos.includes(procno)) {
        allProcnos.push(procno)
      }
    }

    const expnoCheckBox = document.getElementById(`select_expno_${expno}`)
    if (expnoCheckBox.checked) {
      allExpnos.push(expno)
    }
  }

  for (const procno of allProcnos) {
    let valid = true
    let is2d = null
    for (const expno of allExpnos) {
      if (expnoParams.are_2d[`${expno}-${procno}`] === null) {
        valid = false
        break
      }
      if (is2d === null) {
        is2d = expnoParams.are_2d[`${expno}-${procno}`]
      }

      if (expnoParams.are_2d[`${expno}-${procno}`] !== is2d) {
        valid = false
        break
      }
    }
    if (valid) {
      possibleProcnos.push(procno)
    }
  }

  console.log(possibleProcnos)

  // Delete any previously generated element
  const parent1 = document.getElementById('load-procno')
  let parent2 = document.getElementById('select_procno')
  if (parent2 !== null) {
    parent2.remove()
  }
  parent2 = document.createElement('div')
  parent2.id = 'select_procno'
  parent1.appendChild(parent2)
  let first = true
  for (const procno of possibleProcnos) {
    const childLabel = document.createElement('label')
    childLabel.for = 'selected-procno'
    childLabel.innerHTML = `Use procno ${procno}`
    parent2.appendChild(childLabel)
    const childBox = document.createElement('input')
    childBox.type = 'radio'
    childBox.name = 'selected-procno'
    childBox.id = `selected-procno-${procno}`
    childBox.value = procno
    if (first) {
      childBox.checked = true
      first = false
    }
    parent2.appendChild(childBox)
    parent2.appendChild(document.createElement('br'))
  }

  console.log(allExpnos)
}

// On dataset upload
$(function () {
  $('#dataset').change(function () {
    if (document.getElementById('dataset').value.length > 0) {
      console.log('Uploading')

      // Upload file to server
      document.getElementById('select-exp').className = 'hide'
      document.getElementById('preprocess').className = 'hide'
      document.getElementById('onpredict').className = 'hide'
      document.getElementById('preds').className = 'hide'
      document.getElementById('onupload').className = 'appear'

      const formData = new FormData()
      formData.append('dataset', document.getElementById('dataset').files[0])
      $.ajax({
        url: '/upload_dataset',
        cache: false,
        contentType: false,
        processData: false,
        data: formData,
        type: 'POST',
        success: function (response) {
          // Show that the file successfully uploaded
          console.log('Uploaded!')
          // Show that the file successfully uploaded
          document.getElementById('onupload').className = 'hide'
          setTimeout(function () {
            document.getElementById('select-exp').className = 'appear'
          }, 600)

          const parent = document.getElementById('load-expnos')
          console.log(response.params)

          for (const [expno, procnos] of Object.entries(response.params.procnos)) {
            const childLabel = document.createElement('label')
            childLabel.className = 'expno_selection'
            childLabel.for = `select_expno_${expno}`
            childLabel.innerHTML = `Include expno ${expno}: `
            parent.appendChild(childLabel)

            const childBox = document.createElement('input')
            childBox.id = `select_expno_${expno}`
            childBox.type = 'checkbox'
            childBox.checked = HasValidProcno(response.params.are_2d, expno, procnos)
            childBox.disabled = !HasValidProcno(response.params.are_2d, expno, procnos)
            childBox.classname = 'expno_selection_checkbox'
            childBox.onchange = function () { GetPossibleProcnos() }
            parent.appendChild(childBox)
            parent.appendChild(document.createElement('br'))
          }

          expnoParams = response.params

          GetPossibleProcnos()
        },
        error: function (error) {
          console.log(error)
          document.getElementById('errormsg').innerHTML = error.responseText

          document.getElementById('preprocess').className = 'hide'
          document.getElementById('onpredict').className = 'hide'
          document.getElementById('preds').className = 'hide'
          document.getElementById('onupload').className = 'hide'

          document.getElementById('errormsg').className = 'appear'

          setTimeout(function () {
            document.getElementById('errormsg').className = 'hide'
          }, 10000)
          document.getElementById('dataset').value = ''
        }
      })
    }
  })
})

function GetSelectedExpnos () {
  const expnos = []
  for (const [expno, procnos] of Object.entries(expnoParams.procnos)) {
    const box = document.getElementById(`select_expno_${expno}`)
    if (box !== null) {
      if (box.checked) {
        expnos.push(expno)
      }
    }
  }
  return expnos
}

function GetSelectedProcno () {
  let procno = null
  const elems = document.getElementsByName('selected-procno')
  for (const elem of elems) {
    if (elem.checked) {
      procno = elem.id
    }
  }
  return procno
}

function LoadDataset () {
  document.getElementById('preprocess').className = 'hide'
  document.getElementById('onpredict').className = 'hide'
  document.getElementById('preds').className = 'hide'
  document.getElementById('onload').className = 'appear'

  const formData = new FormData()
  formData.append('expnos', GetSelectedExpnos())
  formData.append('procno', GetSelectedProcno())
  $.ajax({
    url: '/load_dataset',
    cache: false,
    contentType: false,
    processData: false,
    data: formData,
    type: 'POST',
    success: function (response) {
      // Show that the file successfully uploaded
      document.getElementById('onload').className = 'hide'
      setTimeout(function () {
        document.getElementById('preprocess').className = 'appear'
      }, 600)

      // Show spectra
      console.log(response)
      if (response.spectra[0].ppm2.length === 0) {
        document.getElementById('acqu2').disabled = true
      }

      const n = response.spectra.length
      for (let i = 0; i < response.spectra.length; i++) {
        const d = document.createElement('div')
        d.className = 'spectrum'

        let child = document.createElement('div')
        child.className = 'spectrum_fig'
        child.id = `spectrum_fig${i + 1}`
        let child2 = document.createElement('div')
        child2.className = 'numbertext'
        child2.innerHTML = `${i + 1} / ${n}`
        child.appendChild(child2)
        d.appendChild(child)

        child = document.createElement('div')
        child.className = 'spectrum_title'
        child2 = document.createElement('h4')
        child2.innerHTML = 'Spectrum information'
        child.appendChild(child2)
        child2 = document.createElement('p')
        child2.innerHTML = response.spectra[i].title
        child.appendChild(child2)
        d.appendChild(child)

        child = document.createElement('div')
        child.className = 'spectrum_params'

        child2 = document.createElement('label')
        child2.for = `select_${i + 1}`
        child2.innerHTML = 'Include spectrum'
        child.appendChild(child2)
        child.appendChild(document.createElement('br'))
        child2 = document.createElement('input')
        child2.id = `select_${i + 1}`
        child2.type = 'checkbox'
        child2.value = '1'
        child2.checked = true
        child2.className = 'include-checkbox'
        child.appendChild(child2)
        child.appendChild(document.createElement('br'))
        child.appendChild(document.createElement('br'))

        child2 = document.createElement('label')
        child2.for = `mas_fig${i + 1}`
        child2.innerHTML = 'MAS rate [kHz]'
        child.appendChild(child2)
        child.appendChild(document.createElement('br'))

        child2 = document.createElement('input')
        child2.id = `mas_fig${i + 1}`
        child2.type = 'number'
        child2.min = '1'
        child2.value = response.spectra[i].wr
        child2.innerHTML = 'MAS rate [kHz]'
        child2.className = 'masr'
        child.appendChild(child2)

        d.appendChild(child)

        document.getElementById('spectra').appendChild(d)
      }

      data = response.spectra

      console.log(data)

      ShowSpectrum(0)
      UpdateSelMas()
    },
    error: function (error) {
      // Show that the file successfully uploaded
      document.getElementById('onload').className = 'hide'
      console.log(error)
      document.getElementById('errormsg').innerHTML = error.responseText

      document.getElementById('preprocess').className = 'hide'
      document.getElementById('onpredict').className = 'hide'
      document.getElementById('preds').className = 'hide'
      document.getElementById('onupload').className = 'hide'

      document.getElementById('errormsg').className = 'appear'

      setTimeout(function () {
        document.getElementById('errormsg').className = 'hide'
      }, 10000)
      document.getElementById('dataset').value = ''
    }
  })
}

function RunPrediction () {
  document.getElementById('preds').className = 'hide'
  document.getElementById('onpredict').className = 'appear'

  const formData = new FormData()

  for (let i = 0; i < data.length; i++) {
    data[i].wr = document.getElementById(`mas_fig${i + 1}`).value
    data[i].include = document.getElementById(`select_${i + 1}`).checked
  }

  formData.append('data', JSON.stringify(data))
  formData.append('rl', document.getElementById('rangel').value)
  formData.append('rr', document.getElementById('ranger').value)
  formData.append('sens', document.getElementById('sens').value)
  formData.append('acqu2', document.getElementById('acqu2').checked)
  formData.append('units', unit)

  if (predicting) {
    console.log('Already running prediction!')
  } else {
    predicting = true
    $.ajax({
      url: '/run_prediction',
      cache: false,
      contentType: false,
      processData: false,
      data: formData,
      type: 'POST',
      success: function (response) {
        preds = response.preds

        PlotPred(document.getElementById('final-pred'), predInd = preds.preds.length - 1)

        PlotPredAll(document.getElementById('final-pred-all-specs'), 0.1)

        PlotPred(document.getElementById('evolving-pred'), predInd = 0)

        document.getElementById('onpredict').className = 'hide'
        setTimeout(function () {
          document.getElementById('preds').className = 'appear'
        }, 600)

        predicting = false

        const dlData = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(preds))
        const dlElem = document.getElementById('download')
        dlElem.setAttribute('href', dlData)
        dlElem.setAttribute('download', 'preds.json')
      },
      error: function (error) {
        predicting = false
        console.log(error)
      }
    })
  }
}
