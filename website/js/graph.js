/* =========================================================
   graph.js  –  D3 force-directed propagation graph renderer
   ========================================================= */

class GraphRenderer {
  constructor(containerId) {
    this.containerId = containerId;
    this.simulation = null;
    this.svg = null;
    this.width = 0;
    this.height = 0;
  }

  _init() {
    const container = document.getElementById(this.containerId);
    container.innerHTML = '';
    const rect = container.getBoundingClientRect();
    this.width  = rect.width  || 560;
    this.height = rect.height || 480;

    this.svg = d3.select(`#${this.containerId}`)
      .append('svg')
      .attr('width',  this.width)
      .attr('height', this.height);

    // Gradient defs
    const defs = this.svg.append('defs');

    const fakeGrad = defs.append('radialGradient').attr('id', 'fakeGrad');
    fakeGrad.append('stop').attr('offset', '0%').attr('stop-color', '#ff6b6b');
    fakeGrad.append('stop').attr('offset', '100%').attr('stop-color', '#c0392b');

    const realGrad = defs.append('radialGradient').attr('id', 'realGrad');
    realGrad.append('stop').attr('offset', '0%').attr('stop-color', '#2ecc71');
    realGrad.append('stop').attr('offset', '100%').attr('stop-color', '#1a8f4f');

    const rootGrad = defs.append('radialGradient').attr('id', 'rootGrad');
    rootGrad.append('stop').attr('offset', '0%').attr('stop-color', '#f9ca24');
    rootGrad.append('stop').attr('offset', '100%').attr('stop-color', '#f0932b');

    // Glow filter
    const filter = defs.append('filter').attr('id', 'glow');
    filter.append('feGaussianBlur').attr('stdDeviation', '3.5').attr('result', 'coloredBlur');
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
  }

  render(graphData, label) {
    if (this.simulation) this.simulation.stop();
    this._init();

    const nodes = graphData.nodes.map(n => ({ ...n }));
    const links = graphData.links.map(l => ({
      source: typeof l.source === 'object' ? l.source.id : l.source,
      target: typeof l.target === 'object' ? l.target.id : l.target,
    }));

    const nodeById = new Map(nodes.map(n => [n.id, n]));
    const resolvedLinks = links.map(l => ({
      source: nodeById.get(l.source),
      target: nodeById.get(l.target),
    })).filter(l => l.source && l.target);

    // Degree map for sizing
    const degree = new Map(nodes.map(n => [n.id, 0]));
    links.forEach(({ source, target }) => {
      degree.set(source, (degree.get(source) || 0) + 1);
      degree.set(target, (degree.get(target) || 0) + 1);
    });

    const maxDeg = Math.max(...degree.values(), 1);

    const nodeRadius = n => {
      if (n.group === 'root') return 14;
      const d = degree.get(n.id) || 1;
      return 5 + (d / maxDeg) * 10;
    };

    const nodeColor = n => {
      if (n.group === 'root') return 'url(#rootGrad)';
      return label === 1 ? 'url(#fakeGrad)' : 'url(#realGrad)';
    };

    const glowColor = label === 1 ? '#ff6b6b' : '#2ecc71';

    // Simulation
    this.simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(resolvedLinks).id(d => d.id).distance(50).strength(0.8))
      .force('charge', d3.forceManyBody().strength(-180))
      .force('center', d3.forceCenter(this.width / 2, this.height / 2))
      .force('collide', d3.forceCollide().radius(d => nodeRadius(d) + 4));

    // Draw edges
    const link = this.svg.append('g').selectAll('line')
      .data(resolvedLinks)
      .join('line')
      .attr('stroke', label === 1 ? 'rgba(255,107,107,0.25)' : 'rgba(46,204,113,0.25)')
      .attr('stroke-width', 1.2);

    // Draw nodes
    const node = this.svg.append('g').selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', d => nodeRadius(d))
      .attr('fill', d => nodeColor(d))
      .attr('filter', d => d.group === 'root' ? 'url(#glow)' : null)
      .attr('stroke', d => d.group === 'root' ? '#f9ca24' : 'rgba(255,255,255,0.15)')
      .attr('stroke-width', d => d.group === 'root' ? 2.5 : 1)
      .attr('opacity', 0)
      .call(
        d3.drag()
          .on('start', (event, d) => {
            if (!event.active) this.simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => {
            if (!event.active) this.simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      );

    // Tooltip
    const tooltip = d3.select('body').selectAll('#graph-tooltip').data([1])
      .join('div').attr('id', 'graph-tooltip');

    node.on('mouseover', (event, d) => {
      tooltip
        .style('opacity', 1)
        .style('left', (event.pageX + 12) + 'px')
        .style('top',  (event.pageY - 28) + 'px')
        .html(`
          <span class="tt-role">${d.group}</span>
          <span class="tt-info">depth ${d.depth} &middot; degree ${degree.get(d.id) || 0}</span>
        `);
      d3.select(event.currentTarget)
        .attr('filter', 'url(#glow)')
        .attr('stroke-width', 2.5);
    }).on('mouseleave', (event, d) => {
      tooltip.style('opacity', 0);
      d3.select(event.currentTarget)
        .attr('filter', d.group === 'root' ? 'url(#glow)' : null)
        .attr('stroke-width', d.group === 'root' ? 2.5 : 1);
    }).on('click', (event, d) => {
      // Pulse animation on clicked node
      d3.select(event.currentTarget)
        .transition().duration(120).attr('r', nodeRadius(d) * 1.5)
        .transition().duration(200).attr('r', nodeRadius(d));
      if (typeof window.onNodeClick === 'function') {
        window.onNodeClick(d, degree.get(d.id) || 0);
      }
    });


    // Fade-in animation
    node.transition().duration(600).delay((_, i) => i * 6).attr('opacity', 1);

    this.simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      node
        .attr('cx', d => Math.max(nodeRadius(d), Math.min(this.width  - nodeRadius(d), d.x)))
        .attr('cy', d => Math.max(nodeRadius(d), Math.min(this.height - nodeRadius(d), d.y)));
    });

    // Stats overlay
    this._drawStats(nodes.length, resolvedLinks.length, glowColor);
  }

  _drawStats(numNodes, numEdges, color) {
    const g = this.svg.append('g').attr('transform', 'translate(14,14)');
    const pill = (y, text) => {
      const bg = g.append('rect')
        .attr('y', y).attr('rx', 8).attr('ry', 8)
        .attr('height', 22).attr('fill', 'rgba(10,14,26,0.7)')
        .attr('stroke', color).attr('stroke-width', 1).attr('opacity', 0.85);
      const t = g.append('text')
        .attr('y', y + 15).attr('x', 10)
        .attr('fill', '#e2e8f0').attr('font-size', 11)
        .attr('font-family', 'Inter, sans-serif')
        .text(text);
      const w = t.node().getComputedTextLength() + 20;
      bg.attr('width', w);
    };
    pill(0,  `${numNodes} nodes`);
    pill(28, `${numEdges} edges`);
  }

  destroy() {
    if (this.simulation) this.simulation.stop();
  }
}
