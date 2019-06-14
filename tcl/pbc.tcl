#execute: vmd -dispdev text -e pbc.tcl -args CODE FILELIST > pbc_log.out 
# e.g: CODE = 6b73B
# e.g: FILELIST = {'../4N6HA/results/namd/step7.12_production.dcd ../4N6HA/results/namd/step7.13_production.dcd '}

package require pbctools
set code [lindex $argv 0]
set files [lindex $argv 1]
set filelist [split [lindex $files 0]]

mol new ../../$code/results/step5_assembly.xplor_ext.psf type psf
mol off top
set sel [atomselect top all]
set nf [llength $filelist]

for { set i 0 } { $i <= $nf } { incr i } {
    set current_file [lindex $filelist $i]
    set out_file $current_file-pbc.dcd
    
    animate read dcd $current_file beg 0 end -1 waitfor all
    pbc wrap -all -compound res -center com -centersel protein 

    animate write dcd $out_file
    animate delete all
}
puts 'DONE'
quit 
