d3.flowchart = function()
{
    var flowchart = {},
        node_width = 24 ,
        node_height = 2 ,
        size = [1, 1] ,
        sort_type = "fitness" ,
        scale_type = "scaled" ,
        nodes = [],
        links = [];


    flowchart.node_width = function( _ )
    {
        if( !arguments.length ) return node_width;
        node_width = +_;
        return flowchart;
    };

    flowchart.node_padding = function( _ )
    {
        if( !arguments.length ) return node_padding;
        node_padding = _;
        return flowchart;
    };

    flowchart.node_height = function( _ )
    {
        if( !arguments.length ) return node_height;
        node_height = _;
        return flowchart;
    }

    flowchart.deltax = function( _ )
    {
        if( !arguments.length ) return deltax;
        deltax = _;
        return flowchart;
    }

    flowchart.sort_type = function( _ )
    {
        if( !arguments.length ) return sort_type;
        sort_type = _;
        return flowchart;
    }

    flowchart.scale_type = function( _ )
    {
        if( !arguments.length ) return scale_type;
        scale_type = _;
        return flowchart;
    }

    flowchart.nodes = function( _ )
    {
        if( !arguments.length ) return nodes;
        nodes = _;
        return flowchart;
    };

    flowchart.links = function( _ )
    {
        if( !arguments.length ) return links;
        links = _;
        return flowchart;
    };

    flowchart.link = function()
    {
        var curvature = .5;

        function link( d )
        {
            var x0 = d.source.x + d.source.dx ,
            x1 = d.target.x ,
            xi = d3.interpolateNumber( x0 , x1 ) ,
            x2 = xi( curvature ) ,
            x3 = xi(1 - curvature) ,
            y0 = d.source.y + d.source.dy / 2.0 ,
            y1 = d.target.y + d.target.dy / 2.0 ;
            return "M" + x0 + "," + y0
                + "C" + x2 + "," + y0
                + " " + x3 + "," + y1
                + " " + x1 + "," + y1;
        }

        link.curvature = function( _ )
        {
            if( !arguments.length ) return curvature;
            curvature = +_;
            return link;
        };

        return link;
    }

    flowchart.size = function( _ )
    {
        if( !arguments.length ) return size;
        size = _;
        return flowchart;
    };

    flowchart.init = function()
    {
        compute_node_links();
        compute_ancestors();
    }

    flowchart.layout = function()
    {
        compute_node_positions();
    }

    function compute_node_links()
    {
        var i = 0;
        nodes.forEach( function( node ) {
            node.source_links = [];
            node.target_links = [];
            node.key = "node-" + i;
            i += 1;
            } );
        
        i = 0;
        links.forEach( function( link ) {
            var source = link.source,
                target = link.target;
            if (typeof source === "number") source = link.source = nodes[link.source];
            if (typeof target === "number") target = link.target = nodes[link.target];
            source.source_links.push( link );
            target.target_links.push( link );
            link.dy = node_height;
            link.key = "link-" + i;
            i += 1;
            } );
    }

    var walk_ancestors = function ( node , current_node , generations )
    {
        current_node.target_links.forEach( function( link ) {
            link.ancestor_of.push( node.key );
            } );
        if( generations > 1 )
        {
            current_node.target_links.forEach( function( link ) {
                link.source.ancestor_of.push( node.key );
                walk_ancestors( node , link.source , generations - 1 );
                } );
        }
    }

    function compute_ancestors()
    {
        links.forEach( function( link ) {
            link.ancestor_of = []; } );
        nodes.forEach( function( node ) {
            node.ancestor_of = []; } );
        nodes.forEach( function( node ) {
            walk_ancestors( node , node , 12 ); } );
    }

    function value( d , key ) { return d[ key ]; }

    function sort_function()
    {
        return function( d1 , d2 ) { return ( d1[ sort_type ] > d2[ sort_type ] ); };
    }

    function compute_node_positions()
    {
        console.log( "Updating flowchart with sort type \"" + sort_type + "\" and scale type \"" + scale_type + "\"")
        var nested_nodes = null;
        if( sort_type != "unsorted" )
        {

            nested_nodes = d3.nest()
                .key( function( d ) { return d.generation;} )
                .sortValues( sort_function() )
                .entries( nodes );
        }
        else
        {
            nested_nodes = d3.nest()
                .key( function( d ) { return d.generation;} )
                .entries( nodes );
        }

        var n = nested_nodes.length;
        deltax = ( size[0] - n * node_width ) / (n-1);
        for( var i=0 ; i<n ; ++i )
        {
            for( var j=0 ; j<nested_nodes[i].values.length ; ++j )
            {
                var node = nested_nodes[i].values[j];
                node.x = i * ( node_width + deltax );
                if( scale_type == "unscaled" || sort_type == "unsorted" )
                {
                    node.y = node_height + j * size[1] / nested_nodes[i].values.length;
                }
                else
                {
                    node.y = node[ sort_type ] * size[1];
                }
                node.dx = node_width;
                node.dy = node_height;
            }
        }
    }

    return flowchart;
}
