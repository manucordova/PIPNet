
var spectrum_index = 0;
var predicting = false;

var pred_idx = 0;

var data;

var preds;

// Next/previous controls
function change_spectrum(n) {
  show_spectrum(spectrum_index += n);
}

function show_spectrum(n) {
  var i;
  var spectra = document.getElementsByClassName("spectrum");

  if (n >= spectra.length) {
    spectrum_index = 0;
  } else if (n < 0) {
    spectrum_index = spectra.length - 1;
  }

  for (i = 0; i < spectra.length; i++) {
      spectra[i].style.display = "none";
  }
  spectra[spectrum_index].style.display = "block";

  var ymax = 0;
  for (i=0;i<data.length;i++) {
    ymax = Math.max(ymax, Math.max(...data[i].yr))
  }

  var disp = document.getElementById(`spectrum_fig${spectrum_index+1}`);

  var xmin = Number(document.getElementById("rangel").value);
  var xmax = Number(document.getElementById("ranger").value);

  function plot_spectrum(spectrum_index) {
    Plotly.newPlot(disp, [{x: data[spectrum_index].x, y: data[spectrum_index].yr.map(function(item) {return item/ymax}) }],
                   {xaxis: {showgrid: false, showline: true, range: [xmin, xmax], zeroline: false, title: {text: "Chemical shift [ppm]"}},
                    yaxis: {showgrid: false, showline: true, range: [0, 1.5], zeroline: false, showticklabels: false, title: {text: "Normalized intensity [a.u.]"}},
                    margin: {l: 40, r: 20, t: 20, b: 40}});
  }

  plot_spectrum(spectrum_index);

}

function change_pred(n) {
  plot_pred(document.getElementById("evolving-pred"), pred_idx += n)
}

function plot_pred(elem, n) {

  if (n >= preds.preds.length) {
    pred_idx = 0;
  } else if (n < 0) {
    pred_idx = preds.preds.length - 1;
  }

  var xmin = Number(document.getElementById("rangel").value);
  var xmax = Number(document.getElementById("ranger").value);

  var spec_idx = pred_idx + preds.specs.length - preds.preds.length;

  var pred_y = preds.preds[pred_idx].map(function(item) {return item/Math.max(...preds.preds[pred_idx])})
  var pred_dy = preds.err[pred_idx].map(function(item) {return item/Math.max(...preds.preds[pred_idx])})
  var pred_ymax = pred_y.map((e, i) => e + pred_dy[i])
  var pred_ymin = pred_y.map((e, i) => e - pred_dy[i])

  Plotly.newPlot(elem, [{x: preds.x, y: preds.specs[spec_idx].map(function(item) {return item/Math.max(...preds.specs[spec_idx])}),
                              hoverinfo: "x+y", name: `${preds.wrs[spec_idx]} KHz MAS spectrum`},
                             {x: preds.x, y: pred_y, hoverinfo: "x+y", name: `Predicted PIP spectrum`},
                             {x: preds.x, y: pred_ymax, fill: "tonexty", type: "scatter",
                              mode: "none", fillcolor: "rgba(255, 127, 14, 0)", hoverinfo: "x", showlegend: false},
                             {x: preds.x, y: pred_ymin, fill: "tonexty", type: "scatter",
                              mode: "none", fillcolor: "rgba(255, 127, 14, 0.2)", hoverinfo: "x", name: `Predicted PIP uncertainty`}],
                 {xaxis: {showgrid: false, showline: true, range: [xmin, xmax], zeroline: false, title: {text: "Chemical shift [ppm]"}},
                  yaxis: {showgrid: false, showline: true, range: [0, 1.5], zeroline: false, showticklabels: false, title: {text: "Normalized intensity [a.u.]"}},
                  margin: {l: 40, r: 20, t: 20, b: 40}, legend: {yanchor: "top", y: 0.99, xanchor: "right", x: 0.99}});
}

function plot_pred_all(elem, dy) {

  var ymax = 1e-12;
  for (i=0;i<preds.specs.length;i++) {
    ymax = Math.max(ymax, Math.max(...preds.specs[i]));
  }

  var xmin = Number(document.getElementById("rangel").value);
  var xmax = Number(document.getElementById("ranger").value);

  var specs_to_plot = []
  var c0 = [0, 255, 255];
  var dc = [0, -255, 0];
  var y0 = dy;
  for (i=0;i<preds.specs.length;i++) {

    var col = c0.map((e, k) => e+dc[k]*i/(preds.specs.length-1));

    specs_to_plot.push({x: preds.x, y: preds.specs[i].map(function(item) {return y0+item/ymax}),
                                hoverinfo: "x", name: `${preds.wrs[i]} KHz MAS spectrum`,
                                line: {color: `rgb(${col[0]}, ${col[1]}, ${col[2]})`},
                                showlegend: (i == 0 || i == preds.specs.length-1)});
    y0 += dy;
  }

  var pred_y = preds.preds[preds.preds.length-1].map(function(item) {return item/Math.max(...preds.preds[preds.preds.length-1])})
  var pred_dy = preds.err[preds.preds.length-1].map(function(item) {return item/Math.max(...preds.preds[preds.preds.length-1])})
  var pred_ymax = pred_y.map((e, i) => e + pred_dy[i])
  var pred_ymin = pred_y.map((e, i) => e - pred_dy[i])

  specs_to_plot.push({x: preds.x, y: pred_y, hoverinfo: "x+y", name: `Predicted PIP spectrum`, line: {color: "rgb(255, 127, 14)"}});
  specs_to_plot.push({x: preds.x, y: pred_ymax, fill: "tonexty", type: "scatter",
                      mode: "none", fillcolor: "rgba(255, 127, 14, 0)", hoverinfo: "x", showlegend: false})
  specs_to_plot.push({x: preds.x, y: pred_ymin, fill: "tonexty", type: "scatter",
                      mode: "none", fillcolor: "rgba(255, 127, 14, 0.2)", hoverinfo: "x", name: `Predicted PIP uncertainty`})

  Plotly.newPlot(elem, specs_to_plot,
                 {xaxis: {showgrid: false, showline: true, range: [xmin, xmax], zeroline: false, title: {text: "Chemical shift [ppm]"}},
                  yaxis: {showgrid: false, showline: true, range: [0, 2+y0], zeroline: false, showticklabels: false, title: {text: "Normalized intensity [a.u.]"}},
                  margin: {l: 40, r: 20, t: 20, b: 40}, legend: {yanchor: "top", y: 0.99, xanchor: "right", x: 0.99}});
}

function update_sel_mas() {
  var wr_min = Number(document.getElementById("mas-rangel").value);
  var wr_max = Number(document.getElementById("mas-ranger").value);

  for (i=0;i<data.length;i++) {
    if (Number(document.getElementById(`mas_fig${i+1}`).value) >= wr_min && Number(document.getElementById(`mas_fig${i+1}`).value) <= wr_max) {
      document.getElementById(`select_${i+1}`).checked = true;
    } else {
      document.getElementById(`select_${i+1}`).checked = false;
    }
  }

}

$(function() {
  $("#dataset").change(function() {

    if (document.getElementById("dataset").value.length > 0) {

      console.log("Uploading")

      // Upload file to server
      document.getElementById("preprocess").className = "hide";
      document.getElementById("onpredict").className = "hide";
      document.getElementById("preds").className = "hide";
      setTimeout(function() {
        document.getElementById("onupload").className = "appear";
      }, 600);

      var form_data = new FormData();
      form_data.append("dataset", document.getElementById("dataset").files[0])
      $.ajax({
        url: "/upload_dataset",
        cache: false,
        contentType: false,
        processData: false,
        data: form_data,
        type: "POST",
        success: function(response) {

          // Show that the file successfully uploaded
          document.getElementById("onupload").className = "hide";
          setTimeout(function() {
            document.getElementById("preprocess").className = "appear";
          }, 600);

          // Show spectra

          n = response.spectra.length
          for (i=0;i<response.spectra.length;i++) {
            var d = document.createElement("div");
            d.className = "spectrum";



            var child = document.createElement("div");
            child.className = "spectrum_fig";
            child.id = `spectrum_fig${i+1}`;
            var child2 = document.createElement("div");
            child2.className = "numbertext";
            child2.innerHTML = `${i+1} / ${n}`;
            child.appendChild(child2);
            d.appendChild(child);

            var child = document.createElement("div");
            child.className = "spectrum_title";
            var child2 = document.createElement("h4");
            child2.innerHTML = "Spectrum information";
            child.appendChild(child2)
            var child2 = document.createElement("p");
            child2.innerHTML = response.spectra[i].title;
            child.appendChild(child2);
            d.appendChild(child);

            var child = document.createElement("div");
            child.className = "spectrum_params"

            var child2 = document.createElement("label");
            child2.for = `select_${i+1}`;
            child2.innerHTML = "Include spectrum";
            child.appendChild(child2);
            child.appendChild(document.createElement("br"));
            var child2 = document.createElement("input");
            child2.id = `select_${i+1}`;
            child2.type = "checkbox";
            child2.value = "1";
            child2.checked = true;
            child2.className = "include-checkbox"
            child.appendChild(child2);
            child.appendChild(document.createElement("br"));
            child.appendChild(document.createElement("br"));

            var child2 = document.createElement("label");
            child2.for = `mas_fig${i+1}`;
            child2.innerHTML = "MAS rate [kHz]";
            child.appendChild(child2);
            child.appendChild(document.createElement("br"));

            var child2 = document.createElement("input");
            child2.id = `mas_fig${i+1}`;
            child2.type = "number";
            child2.min = "1";
            child2.value = response.spectra[i].wr;
            child2.innerHTML = "MAS rate [kHz]";
            child2.className = "masr"
            child.appendChild(child2);

            d.appendChild(child);

            document.getElementById("spectra").appendChild(d);
          }

          data = response.spectra;

          show_spectrum(0);

          update_sel_mas();

        },
        error: function(error) {

          console.log(error)

          document.getElementById("errormsg").innerHTML = error.responseText;

          setTimeout(function() {
            document.getElementById("preprocess").className = "hide";
            document.getElementById("onpredict").className = "hide";
            document.getElementById("preds").className = "hide";
            document.getElementById("onupload").className = "hide";
          }, 1000);

          document.getElementById("errormsg").className = "appear";

          setTimeout(function() {
            document.getElementById("errormsg").className = "hide";
          }, 3000);
          document.getElementById("dataset").value = "";
        }
      });

    }
  });
});

function run_prediction() {

  document.getElementById("preds").className = "hide";
  setTimeout(function() {
    document.getElementById("onpredict").className = "appear";
  }, 600);

  var form_data = new FormData();

  for (var i=0;i<data.length;i++) {
    data[i].wr = document.getElementById(`mas_fig${i+1}`).value;
    data[i].include = document.getElementById(`select_${i+1}`).checked;
  }

  form_data.append("data", JSON.stringify(data))
  form_data.append("rl", document.getElementById("rangel").value);
  form_data.append("rr", document.getElementById("ranger").value);
  form_data.append("sens", document.getElementById("sens").value);

  if (predicting) {
    console.log("Already running prediction!")
  } else {
    predicting = true;
    $.ajax({
      url: "/run_prediction",
      cache: false,
      contentType: false,
      processData: false,
      data: form_data,
      type: "POST",
      success: function(response) {

        predicting = false;

        preds = response.preds;

        plot_pred(document.getElementById("final-pred"), pred_idx=preds.preds.length-1);

        plot_pred_all(document.getElementById("final-pred-all-specs"), 0.1);

        plot_pred(document.getElementById("evolving-pred"), pred_idx=0);

        document.getElementById("onpredict").className = "hide";
        setTimeout(function() {
          document.getElementById("preds").className = "appear";
        }, 600);

        var dl_data = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(preds));
        var dl_elem = document.getElementById("download");
        dl_elem.setAttribute("href", dl_data);
        dl_elem.setAttribute("download", "preds.json");

      },
      error: function(error) {
        predicting = false;
        console.log(error);
      }
    });
  }


}
